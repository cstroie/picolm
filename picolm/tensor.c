#include "tensor.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <pthread.h>
#endif

/* ---- Scratch buffer (kept for dequantize_row in model.c) ---- */

static float *scratch_buf = NULL;
static int    scratch_size = 0;

void tensor_init_scratch(float *buf, int size) {
    scratch_buf  = buf;
    scratch_size = size;
}

/* ================================================================
 * Matmul task descriptor
 * ================================================================ */

typedef struct {
    float       *out;
    const float *x;
    const char  *W;
    size_t       row_bytes;
    int          n;
    int          start;
    int          end;
    gguf_type_t  qtype;
} matmul_task_t;

static void run_task(matmul_task_t *t) {
    for (int i = t->start; i < t->end; i++) {
        /* Prefetch the next row to hide mmap page-fault latency */
#if !defined(_WIN32) && defined(__GNUC__)
        if (i + 1 < t->end)
            __builtin_prefetch(t->W + (size_t)(i + 1) * t->row_bytes, 0, 1);
#endif
        t->out[i] = vec_dot(t->W + (size_t)i * t->row_bytes,
                            t->x, t->n, t->qtype);
    }
}

/* ================================================================
 * Persistent thread pool (Linux/macOS)
 *
 * Each background worker blocks on its own mutex+condvar waiting for
 * work. The main thread assigns a task, sets state=RUNNING, and
 * signals the condvar. The worker executes run_task(), sets
 * state=IDLE, and signals back. No pthread_create/join overhead after
 * pool startup — zero OS calls per matmul on the fast path.
 *
 * Windows falls back to the original create-per-call approach to avoid
 * adding a second OS-specific thread API.
 * ================================================================ */

static int n_threads = 1;

int tensor_get_threads(void) { return n_threads; }

#ifndef _WIN32

#define WORKER_IDLE     0
#define WORKER_RUNNING  1
#define WORKER_SHUTDOWN (-1)

typedef struct {
    pthread_t       tid;
    pthread_mutex_t mu;
    pthread_cond_t  cv;
    matmul_task_t   task;
    volatile int    state;
} pool_worker_t;

static pool_worker_t g_workers[MAX_THREADS];
static int           g_pool_size = 0;

static void *pool_loop(void *arg) {
    pool_worker_t *w = (pool_worker_t *)arg;
    for (;;) {
        pthread_mutex_lock(&w->mu);
        while (w->state == WORKER_IDLE)
            pthread_cond_wait(&w->cv, &w->mu);
        int s = w->state;
        pthread_mutex_unlock(&w->mu);

        if (s == WORKER_SHUTDOWN) break;

        run_task(&w->task);

        pthread_mutex_lock(&w->mu);
        w->state = WORKER_IDLE;
        pthread_cond_signal(&w->cv);
        pthread_mutex_unlock(&w->mu);
    }
    return NULL;
}

static void pool_resize(int nt) {
    /* Shut down existing background workers */
    for (int i = 1; i < g_pool_size; i++) {
        pthread_mutex_lock(&g_workers[i].mu);
        g_workers[i].state = WORKER_SHUTDOWN;
        pthread_cond_signal(&g_workers[i].cv);
        pthread_mutex_unlock(&g_workers[i].mu);
        pthread_join(g_workers[i].tid, NULL);
        pthread_mutex_destroy(&g_workers[i].mu);
        pthread_cond_destroy(&g_workers[i].cv);
    }
    g_pool_size = 0;

    /* Spawn new background workers (index 0 = main thread slot, not spawned) */
    for (int i = 1; i < nt; i++) {
        pool_worker_t *w = &g_workers[i];
        pthread_mutex_init(&w->mu, NULL);
        pthread_cond_init(&w->cv, NULL);
        w->state = WORKER_IDLE;
        pthread_create(&w->tid, NULL, pool_loop, w);
    }
    g_pool_size = nt;
}

#endif /* !_WIN32 */

void tensor_set_threads(int t) {
    if (t < 1) t = 1;
    if (t > MAX_THREADS) t = MAX_THREADS;
    n_threads = t;
#ifndef _WIN32
    pool_resize(t);
#endif
}

/* ================================================================
 * matmul: out[d] = W[d × n] × x[n]   (quantized rows)
 * ================================================================ */

void matmul(float *out, const float *x, const void *W, int n, int d,
            gguf_type_t qtype) {
    size_t row_bytes = gguf_type_row_size(qtype, n);
    const char *wptr = (const char *)W;

#ifdef _WIN32
    /* Windows: original create-per-call approach */
    if (n_threads <= 1 || d < 4) {
        matmul_task_t t = { out, x, wptr, row_bytes, n, 0, d, qtype };
        run_task(&t);
        return;
    }
    int nt = n_threads;
    if (nt > d) nt = d;

    matmul_task_t tasks[MAX_THREADS];
    HANDLE        threads[MAX_THREADS];
    int rows_per = d / nt, extra = d % nt, row = 0;

    for (int t = 0; t < nt; t++) {
        tasks[t].out = out; tasks[t].x = x; tasks[t].W = wptr;
        tasks[t].row_bytes = row_bytes; tasks[t].n = n; tasks[t].qtype = qtype;
        tasks[t].start = row;
        row += rows_per + (t < extra ? 1 : 0);
        tasks[t].end = row;
    }
    for (int t = 1; t < nt; t++)
        threads[t] = CreateThread(NULL, 0,
                         (LPTHREAD_START_ROUTINE)run_task, &tasks[t], 0, NULL);
    run_task(&tasks[0]);
    for (int t = 1; t < nt; t++) {
        WaitForSingleObject(threads[t], INFINITE);
        CloseHandle(threads[t]);
    }

#else
    /* Linux/macOS: persistent thread pool */
    int nt = (g_pool_size > 0) ? g_pool_size : 1;
    if (nt > d) nt = d;
    if (nt <= 1) {
        matmul_task_t t = { out, x, wptr, row_bytes, n, 0, d, qtype };
        run_task(&t);
        return;
    }

    /* Distribute rows evenly across threads */
    int rows_per = d / nt, extra = d % nt, row = 0;
    int starts[MAX_THREADS], ends[MAX_THREADS];
    for (int t = 0; t < nt; t++) {
        starts[t] = row;
        row += rows_per + (t < extra ? 1 : 0);
        ends[t] = row;
    }

    /* Wake background workers 1..nt-1 */
    for (int t = 1; t < nt; t++) {
        pool_worker_t *w = &g_workers[t];
        w->task = (matmul_task_t){ out, x, wptr, row_bytes, n,
                                   starts[t], ends[t], qtype };
        pthread_mutex_lock(&w->mu);
        w->state = WORKER_RUNNING;
        pthread_cond_signal(&w->cv);
        pthread_mutex_unlock(&w->mu);
    }

    /* Main thread handles range 0 */
    matmul_task_t t0 = { out, x, wptr, row_bytes, n, starts[0], ends[0], qtype };
    run_task(&t0);

    /* Wait for all workers to finish */
    for (int t = 1; t < nt; t++) {
        pool_worker_t *w = &g_workers[t];
        pthread_mutex_lock(&w->mu);
        while (w->state == WORKER_RUNNING)
            pthread_cond_wait(&w->cv, &w->mu);
        pthread_mutex_unlock(&w->mu);
    }
#endif
}

/* ================================================================
 * Fast exp approximation used by SIMD SiLU
 *
 * Range reduction: x = k*ln2 + r, k ∈ Z, r ∈ [-ln2/2, ln2/2]
 * Polynomial: P(r) = 1 + r*(1 + r*(1/2 + r*(1/6 + r*(1/24 + r/120))))
 * Scale: exp(x) = 2^k * P(r)  via float exponent-field shift
 *
 * Max relative error: ~7e-4 (5th-order Taylor in reduced range)
 * This is negligible compared to Q4/Q5 quantization noise.
 * ================================================================ */

#ifdef PICOLM_SSE2
static inline __m128 exp_ps(__m128 x) {
    const __m128 lo    = _mm_set1_ps(-88.f);
    const __m128 hi    = _mm_set1_ps( 88.f);
    const __m128 log2e = _mm_set1_ps(1.44269504088896341f);
    const __m128 ln2   = _mm_set1_ps(0.69314718055994531f);
    const __m128 one   = _mm_set1_ps(1.0f);
    const __m128 c5    = _mm_set1_ps(0.00833333333f); /* 1/120 */
    const __m128 c4    = _mm_set1_ps(0.04166666667f); /* 1/24  */
    const __m128 c3    = _mm_set1_ps(0.16666666667f); /* 1/6   */
    const __m128 c2    = _mm_set1_ps(0.50000000000f); /* 1/2   */

    x = _mm_max_ps(_mm_min_ps(x, hi), lo);

    /* k = round(x * log2e) via round-to-nearest convert */
    __m128i k  = _mm_cvtps_epi32(_mm_mul_ps(x, log2e));
    __m128  kf = _mm_cvtepi32_ps(k);
    __m128  r  = _mm_sub_ps(x, _mm_mul_ps(kf, ln2)); /* r ∈ [-ln2/2, ln2/2] */

    /* Horner: P = (((c5*r+c4)*r+c3)*r+c2)*r+1)*r+1 */
    __m128 p = c5;
    p = _mm_add_ps(_mm_mul_ps(p, r), c4);
    p = _mm_add_ps(_mm_mul_ps(p, r), c3);
    p = _mm_add_ps(_mm_mul_ps(p, r), c2);
    p = _mm_add_ps(_mm_mul_ps(p, r), one);
    p = _mm_add_ps(_mm_mul_ps(p, r), one);

    /* Scale by 2^k: insert k into float exponent field */
    __m128i ki = _mm_slli_epi32(_mm_add_epi32(k, _mm_set1_epi32(127)), 23);
    return _mm_mul_ps(p, _mm_castsi128_ps(ki));
}
#endif /* PICOLM_SSE2 */

#ifdef PICOLM_AVX
static inline __m256 exp_avx(__m256 x) {
    const __m256 lo    = _mm256_set1_ps(-88.f);
    const __m256 hi    = _mm256_set1_ps( 88.f);
    const __m256 log2e = _mm256_set1_ps(1.44269504088896341f);
    const __m256 ln2   = _mm256_set1_ps(0.69314718055994531f);
    const __m256 one   = _mm256_set1_ps(1.0f);
    const __m256 c5    = _mm256_set1_ps(0.00833333333f);
    const __m256 c4    = _mm256_set1_ps(0.04166666667f);
    const __m256 c3    = _mm256_set1_ps(0.16666666667f);
    const __m256 c2    = _mm256_set1_ps(0.50000000000f);

    x = _mm256_max_ps(_mm256_min_ps(x, hi), lo);

    __m256i k  = _mm256_cvtps_epi32(_mm256_mul_ps(x, log2e));
    __m256  kf = _mm256_cvtepi32_ps(k);
    __m256  r  = _mm256_sub_ps(x, _mm256_mul_ps(kf, ln2));

    __m256 p = c5;
    p = _mm256_add_ps(_mm256_mul_ps(p, r), c4);
    p = _mm256_add_ps(_mm256_mul_ps(p, r), c3);
    p = _mm256_add_ps(_mm256_mul_ps(p, r), c2);
    p = _mm256_add_ps(_mm256_mul_ps(p, r), one);
    p = _mm256_add_ps(_mm256_mul_ps(p, r), one);

    __m256i ki = _mm256_slli_epi32(_mm256_add_epi32(k, _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(p, _mm256_castsi256_ps(ki));
}
#endif /* PICOLM_AVX */

/* ================================================================
 * SIMD-accelerated basic operations
 * ================================================================ */

void rmsnorm(float *out, const float *x, const float *weight, int size) {
    float ss = 0.0f;

#ifdef PICOLM_NEON
    float32x4_t acc = vdupq_n_f32(0);
    int i = 0;
    for (; i + 3 < size; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        acc = vmlaq_f32(acc, v, v);
    }
    ss = vaddvq_f32_compat(acc);
    for (; i < size; i++) ss += x[i] * x[i];
#elif defined(PICOLM_AVX2) && defined(__FMA__)
    /* AVX2 + FMA: fmadd(v, v, acc) = v*v + acc in one instruction */
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 7 < size; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        acc = _mm256_fmadd_ps(v, v, acc);
    }
    ss = hsum_avx(acc);
    for (; i < size; i++) ss += x[i] * x[i];
#elif defined(PICOLM_AVX)
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 7 < size; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(v, v));
    }
    ss = hsum_avx(acc);
    for (; i < size; i++) ss += x[i] * x[i];
#elif defined(PICOLM_SSE2)
    __m128 acc = _mm_setzero_ps();
    int i = 0;
    for (; i + 3 < size; i += 4) {
        __m128 v = _mm_loadu_ps(x + i);
        acc = _mm_add_ps(acc, _mm_mul_ps(v, v));
    }
    ss = hsum_sse(acc);
    for (; i < size; i++) ss += x[i] * x[i];
#else
    for (int i = 0; i < size; i++) ss += x[i] * x[i];
#endif

    ss = 1.0f / sqrtf(ss / (float)size + 1e-5f);

#ifdef PICOLM_NEON
    float32x4_t scale = vdupq_n_f32(ss);
    i = 0;
    for (; i + 3 < size; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        float32x4_t w = vld1q_f32(weight + i);
        vst1q_f32(out + i, vmulq_f32(vmulq_f32(v, scale), w));
    }
    for (; i < size; i++) out[i] = x[i] * ss * weight[i];
#elif defined(PICOLM_AVX)
    __m256 scale = _mm256_set1_ps(ss);
    i = 0;
    for (; i + 7 < size; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        __m256 w = _mm256_loadu_ps(weight + i);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_mul_ps(v, scale), w));
    }
    for (; i < size; i++) out[i] = x[i] * ss * weight[i];
#elif defined(PICOLM_SSE2)
    __m128 scale = _mm_set1_ps(ss);
    i = 0;
    for (; i + 3 < size; i += 4) {
        __m128 v = _mm_loadu_ps(x + i);
        __m128 w = _mm_loadu_ps(weight + i);
        _mm_storeu_ps(out + i, _mm_mul_ps(_mm_mul_ps(v, scale), w));
    }
    for (; i < size; i++) out[i] = x[i] * ss * weight[i];
#else
    for (int i = 0; i < size; i++) out[i] = x[i] * ss * weight[i];
#endif
}

void softmax(float *x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    float inv = 1.0f / sum;

#ifdef PICOLM_NEON
    float32x4_t inv_v = vdupq_n_f32(inv);
    int i = 0;
    for (; i + 3 < size; i += 4) {
        vst1q_f32(x + i, vmulq_f32(vld1q_f32(x + i), inv_v));
    }
    for (; i < size; i++) x[i] *= inv;
#elif defined(PICOLM_AVX)
    __m256 inv_v = _mm256_set1_ps(inv);
    int i = 0;
    for (; i + 7 < size; i += 8) {
        _mm256_storeu_ps(x + i, _mm256_mul_ps(_mm256_loadu_ps(x + i), inv_v));
    }
    for (; i < size; i++) x[i] *= inv;
#elif defined(PICOLM_SSE2)
    __m128 inv_v = _mm_set1_ps(inv);
    int i = 0;
    for (; i + 3 < size; i += 4) {
        _mm_storeu_ps(x + i, _mm_mul_ps(_mm_loadu_ps(x + i), inv_v));
    }
    for (; i < size; i++) x[i] *= inv;
#else
    for (int i = 0; i < size; i++) x[i] *= inv;
#endif
}

/* ---- AVX RoPE helper: 4 complex pairs per iteration ---- */
#ifdef PICOLM_AVX
static void rope_avx(float *h, int half, const float *cos_pos, const float *sin_pos) {
    int i = 0;
    /* Process 4 complex pairs at a time: [r0,i0,r1,i1,r2,i2,r3,i3]
     * Broadcast cos/sin: [c0,c0,c1,c1,c2,c2,c3,c3] via unpacklo/hi + set_m128.
     * Swap pairs with permute_ps(0xB1) within each 128-bit lane.
     * _mm256_addsub_ps: subtract even lanes, add odd lanes. */
    for (; i + 3 < half; i += 4) {
        __m256 v   = _mm256_loadu_ps(h + i * 2);
        __m128 c4  = _mm_loadu_ps(cos_pos + i);
        __m128 s4  = _mm_loadu_ps(sin_pos + i);
        __m256 cv  = _mm256_set_m128(_mm_unpackhi_ps(c4, c4), _mm_unpacklo_ps(c4, c4));
        __m256 sv  = _mm256_set_m128(_mm_unpackhi_ps(s4, s4), _mm_unpacklo_ps(s4, s4));
        __m256 sw  = _mm256_permute_ps(v, 0xB1); /* swap r,i within each pair */
        _mm256_storeu_ps(h + i * 2,
            _mm256_addsub_ps(_mm256_mul_ps(v, cv), _mm256_mul_ps(sw, sv)));
    }
    for (; i < half; i++) {
        float r = h[i * 2], im = h[i * 2 + 1];
        h[i * 2]     = r * cos_pos[i] - im * sin_pos[i];
        h[i * 2 + 1] = r * sin_pos[i] + im * cos_pos[i];
    }
}
#endif

/* ---- SSE2 RoPE helper: apply rotation to one head ---- */
#if defined(PICOLM_SSE2) && !defined(PICOLM_AVX)
static void rope_sse(float *h, int half, const float *cos_pos, const float *sin_pos) {
    int i = 0;
    /* Process 2 complex pairs at a time: [r0, i0, r1, i1]
     * r_new = r*cos - i*sin,  i_new = r*sin + i*cos
     *
     * With SSE: swap pairs to get [i0, r0, i1, r1], broadcast cos/sin,
     * then use addsub (SSE3) or sign-mask trick (SSE2).
     *   a = v  * cv  = [r0*c0, i0*c0, r1*c1, i1*c1]
     *   b = sv * sv' = [i0*s0, r0*s0, i1*s1, r1*s1]
     *   result[even] = a - b,  result[odd] = a + b  */
#ifdef PICOLM_SSE3
    for (; i + 1 < half; i += 2) {
        __m128 v  = _mm_loadu_ps(h + i * 2);
        __m128 c2 = _mm_unpacklo_ps(_mm_load_ss(cos_pos + i), _mm_load_ss(cos_pos + i + 1));
        __m128 s2 = _mm_unpacklo_ps(_mm_load_ss(sin_pos + i), _mm_load_ss(sin_pos + i + 1));
        __m128 cv = _mm_shuffle_ps(c2, c2, _MM_SHUFFLE(1,1,0,0));
        __m128 sv = _mm_shuffle_ps(s2, s2, _MM_SHUFFLE(1,1,0,0));
        __m128 sw = _mm_shuffle_ps(v,  v,  _MM_SHUFFLE(2,3,0,1));
        _mm_storeu_ps(h + i * 2, _mm_addsub_ps(_mm_mul_ps(v, cv), _mm_mul_ps(sw, sv)));
    }
#else
    const __m128 sign = _mm_set_ps(1.0f, -1.0f, 1.0f, -1.0f);
    for (; i + 1 < half; i += 2) {
        __m128 v  = _mm_loadu_ps(h + i * 2);
        __m128 c2 = _mm_unpacklo_ps(_mm_load_ss(cos_pos + i), _mm_load_ss(cos_pos + i + 1));
        __m128 s2 = _mm_unpacklo_ps(_mm_load_ss(sin_pos + i), _mm_load_ss(sin_pos + i + 1));
        __m128 cv = _mm_shuffle_ps(c2, c2, _MM_SHUFFLE(1,1,0,0));
        __m128 sv = _mm_shuffle_ps(s2, s2, _MM_SHUFFLE(1,1,0,0));
        __m128 sw = _mm_shuffle_ps(v,  v,  _MM_SHUFFLE(2,3,0,1));
        __m128 a  = _mm_mul_ps(v, cv);
        __m128 b  = _mm_mul_ps(_mm_mul_ps(sign, sw), sv);
        _mm_storeu_ps(h + i * 2, _mm_add_ps(a, b));
    }
#endif
    for (; i < half; i++) {
        float r = h[i * 2], im = h[i * 2 + 1];
        h[i * 2]     = r * cos_pos[i] - im * sin_pos[i];
        h[i * 2 + 1] = r * sin_pos[i] + im * cos_pos[i];
    }
}
#endif

/* Rotary position encoding using pre-computed cos/sin tables */
void rope(float *q, float *k, int head_dim, int n_heads, int n_kv_heads,
          const float *cos_pos, const float *sin_pos) {
    int half = head_dim / 2;

    /* Apply RoPE to all query heads */
    for (int h = 0; h < n_heads; h++) {
        float *qh = q + h * head_dim;
#ifdef PICOLM_NEON
        int i = 0;
        for (; i + 3 < half; i += 4) {
            float32x4x2_t qv = vld2q_f32(qh + i * 2);
            float32x4_t cv = vld1q_f32(cos_pos + i);
            float32x4_t sv = vld1q_f32(sin_pos + i);
            float32x4_t new_even = vmlsq_f32(vmulq_f32(qv.val[0], cv), qv.val[1], sv);
            float32x4_t new_odd  = vmlaq_f32(vmulq_f32(qv.val[0], sv), qv.val[1], cv);
            float32x4x2_t result = {{ new_even, new_odd }};
            vst2q_f32(qh + i * 2, result);
        }
        for (; i < half; i++) {
            float q0 = qh[i * 2];
            float q1 = qh[i * 2 + 1];
            qh[i * 2]     = q0 * cos_pos[i] - q1 * sin_pos[i];
            qh[i * 2 + 1] = q0 * sin_pos[i] + q1 * cos_pos[i];
        }
#elif defined(PICOLM_AVX)
        rope_avx(qh, half, cos_pos, sin_pos);
#elif defined(PICOLM_SSE2)
        rope_sse(qh, half, cos_pos, sin_pos);
#else
        for (int i = 0; i < half; i++) {
            float q0 = qh[i * 2];
            float q1 = qh[i * 2 + 1];
            qh[i * 2]     = q0 * cos_pos[i] - q1 * sin_pos[i];
            qh[i * 2 + 1] = q0 * sin_pos[i] + q1 * cos_pos[i];
        }
#endif
    }

    /* Apply RoPE to all KV heads */
    for (int h = 0; h < n_kv_heads; h++) {
        float *kh = k + h * head_dim;
#ifdef PICOLM_AVX
        rope_avx(kh, half, cos_pos, sin_pos);
#elif defined(PICOLM_SSE2)
        rope_sse(kh, half, cos_pos, sin_pos);
#else
        for (int i = 0; i < half; i++) {
            float k0 = kh[i * 2];
            float k1 = kh[i * 2 + 1];
            kh[i * 2]     = k0 * cos_pos[i] - k1 * sin_pos[i];
            kh[i * 2 + 1] = k0 * sin_pos[i] + k1 * cos_pos[i];
        }
#endif
    }
}

/* ================================================================
 * SiLU: silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
 *
 * Vectorized using the fast exp approximation above.
 * Scalar fallback uses the standard expf() for correctness.
 * ================================================================ */

void silu(float *x, int size) {
#ifdef PICOLM_NEON
    /* ARM NEON: process 4 elements at a time using scalar exp (no vexpq on all targets) */
    int i = 0;
    for (; i + 3 < size; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        /* Compute sigmoid for each lane — compiler vectorizes the expf calls */
        float v0 = vgetq_lane_f32(v, 0), v1 = vgetq_lane_f32(v, 1);
        float v2 = vgetq_lane_f32(v, 2), v3 = vgetq_lane_f32(v, 3);
        float s0 = v0 / (1.0f + expf(-v0));
        float s1 = v1 / (1.0f + expf(-v1));
        float s2 = v2 / (1.0f + expf(-v2));
        float s3 = v3 / (1.0f + expf(-v3));
        vst1q_f32(x + i, (float32x4_t){s0, s1, s2, s3});
    }
    for (; i < size; i++) x[i] = x[i] / (1.0f + expf(-x[i]));

#elif defined(PICOLM_AVX)
    /* AVX: 8 elements per iteration using fast exp_avx */
    const __m256 one = _mm256_set1_ps(1.0f);
    int i = 0;
    for (; i + 7 < size; i += 8) {
        __m256 v   = _mm256_loadu_ps(x + i);
        __m256 neg = _mm256_sub_ps(_mm256_setzero_ps(), v); /* -x */
        __m256 e   = exp_avx(neg);                          /* exp(-x) */
        __m256 den = _mm256_add_ps(one, e);                 /* 1 + exp(-x) */
        /* Use fast reciprocal + one Newton-Raphson step for ~24-bit accuracy */
        __m256 rcp = _mm256_rcp_ps(den);
        rcp = _mm256_mul_ps(rcp,
                _mm256_sub_ps(_mm256_set1_ps(2.0f), _mm256_mul_ps(den, rcp)));
        _mm256_storeu_ps(x + i, _mm256_mul_ps(v, rcp));
    }
    for (; i < size; i++) x[i] = x[i] / (1.0f + expf(-x[i]));

#elif defined(PICOLM_SSE2)
    /* SSE2: 4 elements per iteration using fast exp_ps */
    const __m128 one = _mm_set1_ps(1.0f);
    int i = 0;
    for (; i + 3 < size; i += 4) {
        __m128 v   = _mm_loadu_ps(x + i);
        __m128 neg = _mm_sub_ps(_mm_setzero_ps(), v);
        __m128 e   = exp_ps(neg);
        __m128 den = _mm_add_ps(one, e);
        __m128 rcp = _mm_rcp_ps(den);
        rcp = _mm_mul_ps(rcp,
                _mm_sub_ps(_mm_set1_ps(2.0f), _mm_mul_ps(den, rcp)));
        _mm_storeu_ps(x + i, _mm_mul_ps(v, rcp));
    }
    for (; i < size; i++) x[i] = x[i] / (1.0f + expf(-x[i]));

#else
    for (int i = 0; i < size; i++) {
        x[i] = x[i] / (1.0f + expf(-x[i]));
    }
#endif
}

void elemwise_mul(float *out, const float *a, const float *b, int size) {
#ifdef PICOLM_NEON
    int i = 0;
    for (; i + 3 < size; i += 4) {
        vst1q_f32(out + i, vmulq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
    }
    for (; i < size; i++) out[i] = a[i] * b[i];
#elif defined(PICOLM_AVX)
    int i = 0;
    for (; i + 7 < size; i += 8) {
        _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    }
    for (; i < size; i++) out[i] = a[i] * b[i];
#elif defined(PICOLM_SSE2)
    int i = 0;
    for (; i + 3 < size; i += 4) {
        _mm_storeu_ps(out + i, _mm_mul_ps(_mm_loadu_ps(a + i), _mm_loadu_ps(b + i)));
    }
    for (; i < size; i++) out[i] = a[i] * b[i];
#else
    for (int i = 0; i < size; i++) out[i] = a[i] * b[i];
#endif
}

void vec_add(float *a, const float *b, int size) {
#ifdef PICOLM_NEON
    int i = 0;
    for (; i + 3 < size; i += 4) {
        vst1q_f32(a + i, vaddq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
    }
    for (; i < size; i++) a[i] += b[i];
#elif defined(PICOLM_AVX)
    int i = 0;
    for (; i + 7 < size; i += 8) {
        _mm256_storeu_ps(a + i, _mm256_add_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    }
    for (; i < size; i++) a[i] += b[i];
#elif defined(PICOLM_SSE2)
    int i = 0;
    for (; i + 3 < size; i += 4) {
        _mm_storeu_ps(a + i, _mm_add_ps(_mm_loadu_ps(a + i), _mm_loadu_ps(b + i)));
    }
    for (; i < size; i++) a[i] += b[i];
#else
    for (int i = 0; i < size; i++) a[i] += b[i];
#endif
}
