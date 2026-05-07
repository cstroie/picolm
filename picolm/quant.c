#include "quant.h"
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* ================================================================
 * FP16 <-> FP32 conversion (software, no hardware dependency)
 * ================================================================ */

float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;

    if (exp == 0) {
        if (mant == 0) {
            f = sign; /* +/- zero */
        } else {
            /* subnormal: renormalize */
            exp = 1;
            while (!(mant & 0x400)) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3FF;
            f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000 | (mant << 13); /* inf / nan */
    } else {
        f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }

    float result;
    memcpy(&result, &f, sizeof(float));
    return result;
}

uint16_t fp32_to_fp16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));

    uint32_t sign = (bits >> 16) & 0x8000;
    int      exp  = (int)((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = bits & 0x7FFFFF;

    if (((bits >> 23) & 0xFF) == 0) {
        return (uint16_t)sign; /* zero or f32 subnormal -> fp16 zero */
    }
    if (((bits >> 23) & 0xFF) == 0xFF) {
        /* inf / nan */
        return (uint16_t)(sign | 0x7C00 | (mant ? 0x0200 : 0));
    }
    if (exp >= 31) {
        return (uint16_t)(sign | 0x7C00); /* overflow -> inf */
    }
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign; /* too small -> zero */
        /* subnormal fp16 */
        mant |= 0x800000;
        uint32_t shift = (uint32_t)(14 - exp);
        /* round to nearest */
        uint32_t round_bit = 1U << (shift - 1);
        mant = (mant + round_bit) >> shift;
        return (uint16_t)(sign | mant);
    }

    /* round to nearest even */
    mant += 0x00001000; /* bit 12 */
    if (mant & 0x00800000) {
        mant = 0;
        exp++;
        if (exp >= 31) return (uint16_t)(sign | 0x7C00);
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

/* ---- Q4_K helpers ---- */

static inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *sc, uint8_t *mn) {
    if (j < 4) {
        *sc = q[j] & 63;
        *mn = q[j + 4] & 63;
    } else {
        *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *mn = (q[j + 4] >>  4) | ((q[j    ] >> 6) << 4);
    }
}

/* ================================================================
 * Dequantization kernels (scalar — used for embedding lookup etc.)
 * ================================================================ */

void dequantize_row_q4_K(const void *src, float *dst, int n) {
    const block_q4_K *blocks = (const block_q4_K *)src;
    int nb = n / 256;

    for (int i = 0; i < nb; i++) {
        const block_q4_K *b = &blocks[i];
        float d    = fp16_to_fp32(b->d);
        float dmin = fp16_to_fp32(b->dmin);
        const uint8_t *q = b->qs;
        float *y = dst + i * 256;

        int is = 0;
        for (int j = 0; j < 4; j++) {
            uint8_t sc, mn;
            get_scale_min_k4(is, b->scales, &sc, &mn);
            float d1 = d * (float)sc;
            float m1 = dmin * (float)mn;
            get_scale_min_k4(is + 1, b->scales, &sc, &mn);
            float d2 = d * (float)sc;
            float m2 = dmin * (float)mn;

            for (int l = 0; l < 32; l++) {
                y[l]      = d1 * (float)(q[l] & 0xF) - m1;
            }
            for (int l = 0; l < 32; l++) {
                y[l + 32] = d2 * (float)(q[l] >> 4)  - m2;
            }
            y  += 64;
            q  += 32;
            is += 2;
        }
    }
}

void dequantize_row_q3_K(const void *src, float *dst, int n) {
    const block_q3_K *blocks = (const block_q3_K *)src;
    int nb = n / 256;

    for (int i = 0; i < nb; i++) {
        const block_q3_K *b = &blocks[i];
        float d = fp16_to_fp32(b->d);

        int32_t scales[16];
        {
            for (int j = 0; j < 8; j++) {
                scales[j] = (int32_t)(b->scales[j] & 0xF);
            }
            for (int j = 0; j < 8; j++) {
                scales[8 + j] = (int32_t)(b->scales[j] >> 4);
            }
            for (int j = 0; j < 4; j++) {
                scales[2*j]     |= ((b->scales[8 + j]     ) & 3) << 4;
                scales[2*j + 1] |= ((b->scales[8 + j] >> 2) & 3) << 4;
                scales[2*j + 8] |= ((b->scales[8 + j] >> 4) & 3) << 4;
                scales[2*j + 9] |= ((b->scales[8 + j] >> 6) & 3) << 4;
            }
            for (int j = 0; j < 16; j++) {
                scales[j] -= 32;
            }
        }

        const uint8_t *qs    = b->qs;
        const uint8_t *hmask = b->hmask;
        int out_idx = i * 256;

        for (int j = 0; j < 256; j++) {
            int q2 = (qs[j / 4] >> (2 * (j % 4))) & 3;
            int hbit = (hmask[j / 8] >> (j % 8)) & 1;
            int q3 = q2 | (hbit << 2);
            int sb = j / 16;
            dst[out_idx + j] = d * (float)scales[sb] * ((float)q3 - 4.0f);
        }
    }
}

void dequantize_row_q2_K(const void *src, float *dst, int n) {
    const block_q2_K *blocks = (const block_q2_K *)src;
    int nb = n / 256;

    for (int i = 0; i < nb; i++) {
        const block_q2_K *b = &blocks[i];
        float d    = fp16_to_fp32(b->d);
        float dmin = fp16_to_fp32(b->dmin);

        const uint8_t *qs = b->qs;
        int out_idx = i * 256;

        for (int j = 0; j < 256; j++) {
            int q2 = (qs[j / 4] >> (2 * (j % 4))) & 3;
            int sb = j / 16;
            uint8_t sc = b->scales[sb] & 0xF;
            uint8_t mn = b->scales[sb] >> 4;
            dst[out_idx + j] = d * (float)sc * (float)q2 - dmin * (float)mn;
        }
    }
}

void dequantize_row_q6_K(const void *src, float *dst, int n) {
    const block_q6_K *blocks = (const block_q6_K *)src;
    int nb = n / 256;

    for (int i = 0; i < nb; i++) {
        float d = fp16_to_fp32(blocks[i].d);
        const uint8_t *ql = blocks[i].ql;
        const uint8_t *qh = blocks[i].qh;
        const int8_t  *sc = blocks[i].scales;
        float *y = dst + i * 256;

        for (int chunk = 0; chunk < 256; chunk += 128) {
            int is = chunk / 16;
            for (int l = 0; l < 32; l++) {
                int q1 = (int)((ql[l]      & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int q3 = (int)((ql[l]      >> 4)  | (((qh[l] >> 4) & 3) << 4)) - 32;
                int q4 = (int)((ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4)) - 32;
                int is_l = is + (l / 16);
                y[l]      = d * (float)sc[is_l + 0] * (float)q1;
                y[l + 32] = d * (float)sc[is_l + 2] * (float)q2;
                y[l + 64] = d * (float)sc[is_l + 4] * (float)q3;
                y[l + 96] = d * (float)sc[is_l + 6] * (float)q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
        }
    }
}

void dequantize_row_q8_0(const void *src, float *dst, int n) {
    const block_q8_0 *blocks = (const block_q8_0 *)src;
    int nb = n / 32;

    for (int i = 0; i < nb; i++) {
        float d = fp16_to_fp32(blocks[i].d);
        for (int j = 0; j < 32; j++) {
            dst[i * 32 + j] = d * (float)blocks[i].qs[j];
        }
    }
}

void dequantize_row_q4_0(const void *src, float *dst, int n) {
    const block_q4_0 *blocks = (const block_q4_0 *)src;
    int nb = n / 32;

    for (int i = 0; i < nb; i++) {
        float d = fp16_to_fp32(blocks[i].d);
        for (int j = 0; j < 32; j++) {
            uint8_t nibble;
            if (j < 16) {
                nibble = blocks[i].qs[j] & 0xF;
            } else {
                nibble = blocks[i].qs[j - 16] >> 4;
            }
            dst[i * 32 + j] = d * ((float)nibble - 8.0f);
        }
    }
}

void dequantize_row_f16(const void *src, float *dst, int n) {
    const uint16_t *fp16 = (const uint16_t *)src;
    for (int i = 0; i < n; i++) {
        dst[i] = fp16_to_fp32(fp16[i]);
    }
}

void dequantize_row_f32(const void *src, float *dst, int n) {
    memcpy(dst, src, n * sizeof(float));
}

void dequantize_row(const void *src, float *dst, int n, gguf_type_t type) {
    switch (type) {
        case GGUF_TYPE_F32:   dequantize_row_f32(src, dst, n);  break;
        case GGUF_TYPE_F16:   dequantize_row_f16(src, dst, n);  break;
        case GGUF_TYPE_Q4_0:  dequantize_row_q4_0(src, dst, n); break;
        case GGUF_TYPE_Q8_0:  dequantize_row_q8_0(src, dst, n); break;
        case GGUF_TYPE_Q2_K:  dequantize_row_q2_K(src, dst, n); break;
        case GGUF_TYPE_Q3_K:  dequantize_row_q3_K(src, dst, n); break;
        case GGUF_TYPE_Q4_K:  dequantize_row_q4_K(src, dst, n); break;
        case GGUF_TYPE_Q6_K:  dequantize_row_q6_K(src, dst, n); break;
        default:
            fprintf(stderr, "dequantize_row: unsupported type %d\n", type);
            exit(1);
    }
}

/* ---- Type info ---- */

int gguf_type_block_size(gguf_type_t type) {
    switch (type) {
        case GGUF_TYPE_F32:   return 1;
        case GGUF_TYPE_F16:   return 1;
        case GGUF_TYPE_Q4_0:  return 32;
        case GGUF_TYPE_Q4_1:  return 32;
        case GGUF_TYPE_Q5_0:  return 32;
        case GGUF_TYPE_Q5_1:  return 32;
        case GGUF_TYPE_Q8_0:  return 32;
        case GGUF_TYPE_Q8_1:  return 32;
        case GGUF_TYPE_Q2_K:  return 256;
        case GGUF_TYPE_Q3_K:  return 256;
        case GGUF_TYPE_Q4_K:  return 256;
        case GGUF_TYPE_Q5_K:  return 256;
        case GGUF_TYPE_Q6_K:  return 256;
        default: return 0;
    }
}

int gguf_type_quant_size(gguf_type_t type) {
    switch (type) {
        case GGUF_TYPE_F32:   return 4;
        case GGUF_TYPE_F16:   return 2;
        case GGUF_TYPE_Q4_0:  return 18;
        case GGUF_TYPE_Q4_1:  return 20;
        case GGUF_TYPE_Q5_0:  return 22;
        case GGUF_TYPE_Q5_1:  return 24;
        case GGUF_TYPE_Q8_0:  return 34;
        case GGUF_TYPE_Q8_1:  return 40;
        case GGUF_TYPE_Q2_K:  return 84;
        case GGUF_TYPE_Q3_K:  return 110;
        case GGUF_TYPE_Q4_K:  return 144;
        case GGUF_TYPE_Q5_K:  return 176;
        case GGUF_TYPE_Q6_K:  return 210;
        default: return 0;
    }
}

size_t gguf_type_row_size(gguf_type_t type, int n) {
    int bs = gguf_type_block_size(type);
    int qs = gguf_type_quant_size(type);
    if (bs == 0 || qs == 0) return 0;
    return (size_t)(n / bs) * qs;
}

/* ================================================================
 * Fused dequant + dot-product: compute dot(dequant(row), x) without
 * materializing the full dequantized row.
 *
 * Three tiers per format:
 *   1. NEON (ARM Pi 3/4/5)
 *   2. SSE2 (x86 development)
 *   3. Scalar fallback
 * ================================================================ */

/* ---- vec_dot_f32_f32 ---- */

float vec_dot_f32_f32(const void *src, const float *x, int n) {
    const float *w = (const float *)src;

#ifdef PICOLM_NEON
    float32x4_t acc0 = vdupq_n_f32(0);
    float32x4_t acc1 = vdupq_n_f32(0);
    int i = 0;
    for (; i + 7 < n; i += 8) {
        acc0 = vmlaq_f32(acc0, vld1q_f32(w + i),     vld1q_f32(x + i));
        acc1 = vmlaq_f32(acc1, vld1q_f32(w + i + 4), vld1q_f32(x + i + 4));
    }
    float sum = vaddvq_f32_compat(vaddq_f32(acc0, acc1));
    for (; i < n; i++) sum += w[i] * x[i];
    return sum;

#elif defined(PICOLM_AVX)
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 15 < n; i += 16) {
        acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(_mm256_loadu_ps(w + i),     _mm256_loadu_ps(x + i)));
        acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(_mm256_loadu_ps(w + i + 8), _mm256_loadu_ps(x + i + 8)));
    }
    float sum = hsum_avx(_mm256_add_ps(acc0, acc1));
    for (; i < n; i++) sum += w[i] * x[i];
    return sum;

#elif defined(PICOLM_SSE2)
    __m128 acc0 = _mm_setzero_ps();
    __m128 acc1 = _mm_setzero_ps();
    int i = 0;
    for (; i + 7 < n; i += 8) {
        acc0 = _mm_add_ps(acc0, _mm_mul_ps(_mm_loadu_ps(w + i),     _mm_loadu_ps(x + i)));
        acc1 = _mm_add_ps(acc1, _mm_mul_ps(_mm_loadu_ps(w + i + 4), _mm_loadu_ps(x + i + 4)));
    }
    float sum = hsum_sse(_mm_add_ps(acc0, acc1));
    for (; i < n; i++) sum += w[i] * x[i];
    return sum;

#else
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += w[i] * x[i];
    }
    return sum;
#endif
}

/* ---- vec_dot_q4_K_f32 ---- */

float vec_dot_q4_K_f32(const void *src, const float *x, int n) {
    const block_q4_K *blocks = (const block_q4_K *)src;
    int nb = n / 256;
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        const block_q4_K *b = &blocks[i];
        float d    = fp16_to_fp32(b->d);
        float dmin = fp16_to_fp32(b->dmin);
        const uint8_t *q = b->qs;
        const float *xp = x + i * 256;

        int is = 0;
        for (int j = 0; j < 4; j++) {
            uint8_t sc, mn;
            get_scale_min_k4(is, b->scales, &sc, &mn);
            float d1 = d * (float)sc;
            float m1 = dmin * (float)mn;
            get_scale_min_k4(is + 1, b->scales, &sc, &mn);
            float d2 = d * (float)sc;
            float m2 = dmin * (float)mn;

#ifdef PICOLM_NEON
            float32x4_t sum_qx1_v = vdupq_n_f32(0);
            float32x4_t sum_x1_v  = vdupq_n_f32(0);
            float32x4_t sum_qx2_v = vdupq_n_f32(0);
            float32x4_t sum_x2_v  = vdupq_n_f32(0);

            for (int l = 0; l < 32; l += 8) {
                /* Load 8 quantized bytes, extract nibbles */
                uint8x8_t qbytes = vld1_u8(q + l);
                uint8x8_t q_lo_8 = vand_u8(qbytes, vdup_n_u8(0xF));
                uint8x8_t q_hi_8 = vshr_n_u8(qbytes, 4);

                /* Widen to 16-bit */
                uint16x8_t q_lo_16 = vmovl_u8(q_lo_8);
                uint16x8_t q_hi_16 = vmovl_u8(q_hi_8);

                /* First 4 elements */
                float32x4_t qf0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(q_lo_16)));
                float32x4_t xv0 = vld1q_f32(xp + l);
                sum_qx1_v = vmlaq_f32(sum_qx1_v, qf0, xv0);
                sum_x1_v  = vaddq_f32(sum_x1_v, xv0);

                float32x4_t qf0h = vcvtq_f32_u32(vmovl_u16(vget_low_u16(q_hi_16)));
                float32x4_t xv0h = vld1q_f32(xp + l + 32);
                sum_qx2_v = vmlaq_f32(sum_qx2_v, qf0h, xv0h);
                sum_x2_v  = vaddq_f32(sum_x2_v, xv0h);

                /* Next 4 elements */
                float32x4_t qf1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(q_lo_16)));
                float32x4_t xv1 = vld1q_f32(xp + l + 4);
                sum_qx1_v = vmlaq_f32(sum_qx1_v, qf1, xv1);
                sum_x1_v  = vaddq_f32(sum_x1_v, xv1);

                float32x4_t qf1h = vcvtq_f32_u32(vmovl_u16(vget_high_u16(q_hi_16)));
                float32x4_t xv1h = vld1q_f32(xp + l + 32 + 4);
                sum_qx2_v = vmlaq_f32(sum_qx2_v, qf1h, xv1h);
                sum_x2_v  = vaddq_f32(sum_x2_v, xv1h);
            }

            float sum_qx1 = vaddvq_f32_compat(sum_qx1_v);
            float sum_x1  = vaddvq_f32_compat(sum_x1_v);
            float sum_qx2 = vaddvq_f32_compat(sum_qx2_v);
            float sum_x2  = vaddvq_f32_compat(sum_x2_v);
#elif defined(PICOLM_AVX)
            /* AVX: 128-bit nibble extraction (no AVX2 int needed), 256-bit float accumulators */
            __m256 sum_qx1_v = _mm256_setzero_ps();
            __m256 sum_x1_v  = _mm256_setzero_ps();
            __m256 sum_qx2_v = _mm256_setzero_ps();
            __m256 sum_x2_v  = _mm256_setzero_ps();
            const __m128i mask4  = _mm_set1_epi8(0x0F);
            const __m128i zero_i = _mm_setzero_si128();

            for (int l = 0; l < 32; l += 8) {
                __m128i qb  = _mm_loadl_epi64((const __m128i *)(q + l));
                __m128i lo8 = _mm_and_si128(qb, mask4);
                __m128i hi8 = _mm_and_si128(_mm_srli_epi16(qb, 4), mask4);

                __m128i lo16 = _mm_unpacklo_epi8(lo8, zero_i);
                __m128i hi16 = _mm_unpacklo_epi8(hi8, zero_i);

                /* Combine two __m128 → one __m256 of 8 floats */
                __m256 qf_lo = _mm256_set_m128(
                    _mm_cvtepi32_ps(_mm_unpackhi_epi16(lo16, zero_i)),
                    _mm_cvtepi32_ps(_mm_unpacklo_epi16(lo16, zero_i)));
                __m256 qf_hi = _mm256_set_m128(
                    _mm_cvtepi32_ps(_mm_unpackhi_epi16(hi16, zero_i)),
                    _mm_cvtepi32_ps(_mm_unpacklo_epi16(hi16, zero_i)));

                __m256 xv_lo = _mm256_loadu_ps(xp + l);
                __m256 xv_hi = _mm256_loadu_ps(xp + l + 32);

                sum_qx1_v = _mm256_add_ps(sum_qx1_v, _mm256_mul_ps(qf_lo, xv_lo));
                sum_x1_v  = _mm256_add_ps(sum_x1_v,  xv_lo);
                sum_qx2_v = _mm256_add_ps(sum_qx2_v, _mm256_mul_ps(qf_hi, xv_hi));
                sum_x2_v  = _mm256_add_ps(sum_x2_v,  xv_hi);
            }
            float sum_qx1 = hsum_avx(sum_qx1_v);
            float sum_x1  = hsum_avx(sum_x1_v);
            float sum_qx2 = hsum_avx(sum_qx2_v);
            float sum_x2  = hsum_avx(sum_x2_v);
#elif defined(PICOLM_SSE2)
            /* SSE2: lo nibble → group1 (xp+l), hi nibble → group2 (xp+l+32) */
            __m128 sum_qx1_v = _mm_setzero_ps();
            __m128 sum_x1_v  = _mm_setzero_ps();
            __m128 sum_qx2_v = _mm_setzero_ps();
            __m128 sum_x2_v  = _mm_setzero_ps();
            const __m128i mask4 = _mm_set1_epi8(0x0F);
            const __m128i zero_i = _mm_setzero_si128();

            for (int l = 0; l < 32; l += 8) {
                /* Load 8 quantized bytes -> 8 lo + 8 hi nibbles */
                __m128i qb  = _mm_loadl_epi64((const __m128i *)(q + l));
                __m128i lo8 = _mm_and_si128(qb, mask4);
                __m128i hi8 = _mm_and_si128(_mm_srli_epi16(qb, 4), mask4);

                /* Widen uint8 nibbles -> int32 -> float (8 values each) */
                __m128i lo16 = _mm_unpacklo_epi8(lo8, zero_i);
                __m128i hi16 = _mm_unpacklo_epi8(hi8, zero_i);
                __m128 qf_lo0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(lo16, zero_i));
                __m128 qf_lo1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(lo16, zero_i));
                __m128 qf_hi0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(hi16, zero_i));
                __m128 qf_hi1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(hi16, zero_i));

                __m128 xv_lo0 = _mm_loadu_ps(xp + l);
                __m128 xv_lo1 = _mm_loadu_ps(xp + l + 4);
                __m128 xv_hi0 = _mm_loadu_ps(xp + l + 32);
                __m128 xv_hi1 = _mm_loadu_ps(xp + l + 36);

                sum_qx1_v = _mm_add_ps(sum_qx1_v,
                    _mm_add_ps(_mm_mul_ps(qf_lo0, xv_lo0), _mm_mul_ps(qf_lo1, xv_lo1)));
                sum_x1_v  = _mm_add_ps(sum_x1_v,  _mm_add_ps(xv_lo0, xv_lo1));
                sum_qx2_v = _mm_add_ps(sum_qx2_v,
                    _mm_add_ps(_mm_mul_ps(qf_hi0, xv_hi0), _mm_mul_ps(qf_hi1, xv_hi1)));
                sum_x2_v  = _mm_add_ps(sum_x2_v,  _mm_add_ps(xv_hi0, xv_hi1));
            }
            float sum_qx1 = hsum_sse(sum_qx1_v);
            float sum_x1  = hsum_sse(sum_x1_v);
            float sum_qx2 = hsum_sse(sum_qx2_v);
            float sum_x2  = hsum_sse(sum_x2_v);
#else
            float sum_qx1 = 0.0f, sum_x1 = 0.0f;
            float sum_qx2 = 0.0f, sum_x2 = 0.0f;
            for (int l = 0; l < 32; l++) {
                float x_lo = xp[l];
                float x_hi = xp[l + 32];
                sum_qx1 += (float)(q[l] & 0xF) * x_lo;
                sum_x1  += x_lo;
                sum_qx2 += (float)(q[l] >> 4) * x_hi;
                sum_x2  += x_hi;
            }
#endif
            sumf += d1 * sum_qx1 - m1 * sum_x1 + d2 * sum_qx2 - m2 * sum_x2;

            xp += 64;
            q  += 32;
            is += 2;
        }
    }
    return sumf;
}

/* ---- vec_dot_q6_K_f32 ---- */

float vec_dot_q6_K_f32(const void *src, const float *x, int n) {
    const block_q6_K *blocks = (const block_q6_K *)src;
    int nb = n / 256;
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        float d = fp16_to_fp32(blocks[i].d);
        const uint8_t *ql = blocks[i].ql;
        const uint8_t *qh = blocks[i].qh;
        const int8_t  *sc = blocks[i].scales;
        const float *xp = x + i * 256;

        float sums[16] = {0};

/* sign-extend packed int8 → two __m128 floats; used by both AVX and SSE2 paths */
#define Q6K_CONV(qi8, fa, fb) do { \
    __m128i w16 = _mm_srai_epi16(_mm_unpacklo_epi8(zero_i, qi8), 8); \
    fa = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(zero_i, w16), 16)); \
    fb = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(zero_i, w16), 16)); \
} while (0)

#ifdef PICOLM_AVX
        /* AVX: 128-bit integer extraction, 256-bit float accumulators */
        const __m128i mask4  = _mm_set1_epi8(0x0F);
        const __m128i mask3  = _mm_set1_epi8(0x03);
        const __m128i sub32  = _mm_set1_epi8(32);
        const __m128i zero_i = _mm_setzero_si128();

        for (int chunk = 0; chunk < 2; chunk++) {
            int is = chunk * 8;
            const uint8_t *ql_c = ql + chunk * 64;
            const uint8_t *qh_c = qh + chunk * 32;
            const float   *xp_c = xp + chunk * 128;

            for (int half = 0; half < 2; half++) {
                int l0   = half * 16;
                int sidx = is + half;
                __m256 acc1 = _mm256_setzero_ps();
                __m256 acc2 = _mm256_setzero_ps();
                __m256 acc3 = _mm256_setzero_ps();
                __m256 acc4 = _mm256_setzero_ps();

                for (int l = l0; l < l0 + 16; l += 8) {
                    __m128i qla = _mm_loadl_epi64((const __m128i *)(ql_c + l));
                    __m128i qlb = _mm_loadl_epi64((const __m128i *)(ql_c + l + 32));
                    __m128i qhv = _mm_loadl_epi64((const __m128i *)(qh_c + l));

                    __m128i lo_a = _mm_and_si128(qla, mask4);
                    __m128i hi_a = _mm_and_si128(_mm_srli_epi16(qla, 4), mask4);
                    __m128i lo_b = _mm_and_si128(qlb, mask4);
                    __m128i hi_b = _mm_and_si128(_mm_srli_epi16(qlb, 4), mask4);

                    __m128i h01 = _mm_and_si128(qhv, mask3);
                    __m128i h23 = _mm_and_si128(_mm_srli_epi16(qhv, 2), mask3);
                    __m128i h45 = _mm_and_si128(_mm_srli_epi16(qhv, 4), mask3);
                    __m128i h67 = _mm_and_si128(_mm_srli_epi16(qhv, 6), mask3);

                    __m128i q1_i8 = _mm_sub_epi8(_mm_or_si128(lo_a, _mm_slli_epi16(h01, 4)), sub32);
                    __m128i q2_i8 = _mm_sub_epi8(_mm_or_si128(lo_b, _mm_slli_epi16(h23, 4)), sub32);
                    __m128i q3_i8 = _mm_sub_epi8(_mm_or_si128(hi_a, _mm_slli_epi16(h45, 4)), sub32);
                    __m128i q4_i8 = _mm_sub_epi8(_mm_or_si128(hi_b, _mm_slli_epi16(h67, 4)), sub32);

                    __m128 qf1a, qf1b, qf2a, qf2b, qf3a, qf3b, qf4a, qf4b;
                    Q6K_CONV(q1_i8, qf1a, qf1b);
                    Q6K_CONV(q2_i8, qf2a, qf2b);
                    Q6K_CONV(q3_i8, qf3a, qf3b);
                    Q6K_CONV(q4_i8, qf4a, qf4b);

                    acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(_mm256_set_m128(qf1b, qf1a), _mm256_loadu_ps(xp_c + l)));
                    acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(_mm256_set_m128(qf2b, qf2a), _mm256_loadu_ps(xp_c + l + 32)));
                    acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(_mm256_set_m128(qf3b, qf3a), _mm256_loadu_ps(xp_c + l + 64)));
                    acc4 = _mm256_add_ps(acc4, _mm256_mul_ps(_mm256_set_m128(qf4b, qf4a), _mm256_loadu_ps(xp_c + l + 96)));
                }
                sums[sidx + 0] += hsum_avx(acc1);
                sums[sidx + 2] += hsum_avx(acc2);
                sums[sidx + 4] += hsum_avx(acc3);
                sums[sidx + 6] += hsum_avx(acc4);
            }
        }
#elif defined(PICOLM_SSE2)
        /* SSE2: 6-bit values = lo4(ql) | hi2(qh)<<4, biased by 32 */
        const __m128i mask4  = _mm_set1_epi8(0x0F);
        const __m128i mask3  = _mm_set1_epi8(0x03);
        const __m128i sub32  = _mm_set1_epi8(32);
        const __m128i zero_i = _mm_setzero_si128();

        for (int chunk = 0; chunk < 2; chunk++) {
            int is = chunk * 8;
            const uint8_t *ql_c = ql + chunk * 64;
            const uint8_t *qh_c = qh + chunk * 32;
            const float   *xp_c = xp + chunk * 128;

            for (int half = 0; half < 2; half++) { /* half=0 -> sums[+0,2,4,6], half=1 -> [+1,3,5,7] */
                int l0   = half * 16;
                int sidx = is + half;
                __m128 acc1a = _mm_setzero_ps(), acc1b = _mm_setzero_ps();
                __m128 acc2a = _mm_setzero_ps(), acc2b = _mm_setzero_ps();
                __m128 acc3a = _mm_setzero_ps(), acc3b = _mm_setzero_ps();
                __m128 acc4a = _mm_setzero_ps(), acc4b = _mm_setzero_ps();

                for (int l = l0; l < l0 + 16; l += 8) {
                    __m128i qla = _mm_loadl_epi64((const __m128i *)(ql_c + l));
                    __m128i qlb = _mm_loadl_epi64((const __m128i *)(ql_c + l + 32));
                    __m128i qhv = _mm_loadl_epi64((const __m128i *)(qh_c + l));

                    __m128i lo_a = _mm_and_si128(qla, mask4);
                    __m128i hi_a = _mm_and_si128(_mm_srli_epi16(qla, 4), mask4);
                    __m128i lo_b = _mm_and_si128(qlb, mask4);
                    __m128i hi_b = _mm_and_si128(_mm_srli_epi16(qlb, 4), mask4);

                    /* epi16 shifts on qh: avoids byte-lane bleed from epi8 shifts */
                    __m128i h01 = _mm_and_si128(qhv, mask3);
                    __m128i h23 = _mm_and_si128(_mm_srli_epi16(qhv, 2), mask3);
                    __m128i h45 = _mm_and_si128(_mm_srli_epi16(qhv, 4), mask3);
                    __m128i h67 = _mm_and_si128(_mm_srli_epi16(qhv, 6), mask3);

                    __m128i q1_i8 = _mm_sub_epi8(_mm_or_si128(lo_a, _mm_slli_epi16(h01, 4)), sub32);
                    __m128i q2_i8 = _mm_sub_epi8(_mm_or_si128(lo_b, _mm_slli_epi16(h23, 4)), sub32);
                    __m128i q3_i8 = _mm_sub_epi8(_mm_or_si128(hi_a, _mm_slli_epi16(h45, 4)), sub32);
                    __m128i q4_i8 = _mm_sub_epi8(_mm_or_si128(hi_b, _mm_slli_epi16(h67, 4)), sub32);

                    __m128 qf1a, qf1b, qf2a, qf2b, qf3a, qf3b, qf4a, qf4b;
                    Q6K_CONV(q1_i8, qf1a, qf1b);
                    Q6K_CONV(q2_i8, qf2a, qf2b);
                    Q6K_CONV(q3_i8, qf3a, qf3b);
                    Q6K_CONV(q4_i8, qf4a, qf4b);

                    acc1a = _mm_add_ps(acc1a, _mm_mul_ps(qf1a, _mm_loadu_ps(xp_c + l)));
                    acc1b = _mm_add_ps(acc1b, _mm_mul_ps(qf1b, _mm_loadu_ps(xp_c + l + 4)));
                    acc2a = _mm_add_ps(acc2a, _mm_mul_ps(qf2a, _mm_loadu_ps(xp_c + l + 32)));
                    acc2b = _mm_add_ps(acc2b, _mm_mul_ps(qf2b, _mm_loadu_ps(xp_c + l + 36)));
                    acc3a = _mm_add_ps(acc3a, _mm_mul_ps(qf3a, _mm_loadu_ps(xp_c + l + 64)));
                    acc3b = _mm_add_ps(acc3b, _mm_mul_ps(qf3b, _mm_loadu_ps(xp_c + l + 68)));
                    acc4a = _mm_add_ps(acc4a, _mm_mul_ps(qf4a, _mm_loadu_ps(xp_c + l + 96)));
                    acc4b = _mm_add_ps(acc4b, _mm_mul_ps(qf4b, _mm_loadu_ps(xp_c + l + 100)));
                }
                sums[sidx + 0] += hsum_sse(_mm_add_ps(acc1a, acc1b));
                sums[sidx + 2] += hsum_sse(_mm_add_ps(acc2a, acc2b));
                sums[sidx + 4] += hsum_sse(_mm_add_ps(acc3a, acc3b));
                sums[sidx + 6] += hsum_sse(_mm_add_ps(acc4a, acc4b));
            }
        }
#else
        for (int chunk = 0; chunk < 2; chunk++) {
            int is = chunk * 8;
            const uint8_t *ql_c = ql + chunk * 64;
            const uint8_t *qh_c = qh + chunk * 32;
            const float *xp_c = xp + chunk * 128;

            for (int l = 0; l < 16; l++) {
                int q1 = (int)((ql_c[l]      & 0xF) | (((qh_c[l] >> 0) & 3) << 4)) - 32;
                int q2 = (int)((ql_c[l + 32] & 0xF) | (((qh_c[l] >> 2) & 3) << 4)) - 32;
                int q3 = (int)((ql_c[l]      >> 4)  | (((qh_c[l] >> 4) & 3) << 4)) - 32;
                int q4 = (int)((ql_c[l + 32] >> 4)  | (((qh_c[l] >> 6) & 3) << 4)) - 32;
                sums[is + 0] += (float)q1 * xp_c[l];
                sums[is + 2] += (float)q2 * xp_c[l + 32];
                sums[is + 4] += (float)q3 * xp_c[l + 64];
                sums[is + 6] += (float)q4 * xp_c[l + 96];
            }
            for (int l = 16; l < 32; l++) {
                int q1 = (int)((ql_c[l]      & 0xF) | (((qh_c[l] >> 0) & 3) << 4)) - 32;
                int q2 = (int)((ql_c[l + 32] & 0xF) | (((qh_c[l] >> 2) & 3) << 4)) - 32;
                int q3 = (int)((ql_c[l]      >> 4)  | (((qh_c[l] >> 4) & 3) << 4)) - 32;
                int q4 = (int)((ql_c[l + 32] >> 4)  | (((qh_c[l] >> 6) & 3) << 4)) - 32;
                sums[is + 1] += (float)q1 * xp_c[l];
                sums[is + 3] += (float)q2 * xp_c[l + 32];
                sums[is + 5] += (float)q3 * xp_c[l + 64];
                sums[is + 7] += (float)q4 * xp_c[l + 96];
            }
        }
#endif

#undef Q6K_CONV

        for (int j = 0; j < 16; j++) {
            sumf += d * (float)sc[j] * sums[j];
        }
    }
    return sumf;
}

/* ---- Generic dispatch ---- */

float vec_dot(const void *src, const float *x, int n, gguf_type_t type) {
    switch (type) {
        case GGUF_TYPE_Q4_K: return vec_dot_q4_K_f32(src, x, n);
        case GGUF_TYPE_Q6_K: return vec_dot_q6_K_f32(src, x, n);
        case GGUF_TYPE_F32:  return vec_dot_f32_f32(src, x, n);
        default: {
            /* Fallback: dequantize to temp buffer, then dot */
            float tmp[8192];
            float *buf = (n <= 8192) ? tmp : (float *)malloc((size_t)n * sizeof(float));
            dequantize_row(src, buf, n, type);
            float sum = vec_dot_f32_f32(buf, x, n);
            if (buf != tmp) free(buf);
            return sum;
        }
    }
}
