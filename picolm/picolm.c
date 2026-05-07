#include "model.h"
#include "tensor.h"
#include "tokenizer.h"
#include "sampler.h"
#include "grammar.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#include <windows.h>
static double get_time_ms(void) {
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart / (double)freq.QuadPart * 1000.0;
}
#else
#include <sys/time.h>
#include <unistd.h>
static double get_time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
}
#endif

static void usage(const char *prog) {
    fprintf(stderr, "PicoLLM — ultra-lightweight LLM inference engine\n\n");
    fprintf(stderr, "Usage: %s <model.gguf> [options]\n", prog);
    fprintf(stderr, "\nGeneration options:\n");
    fprintf(stderr, "  -p <prompt>    Input prompt (or pipe via stdin)\n");
    fprintf(stderr, "  -n <int>       Max tokens to generate (default: 256)\n");
    fprintf(stderr, "  -t <float>     Temperature (default: 0.8, 0=greedy)\n");
    fprintf(stderr, "  -k <float>     Top-p / nucleus sampling (default: 0.9)\n");
    fprintf(stderr, "  -s <int>       RNG seed (default: 42)\n");
    fprintf(stderr, "  -c <int>       Context length override\n");
    fprintf(stderr, "  -j <int>       Number of threads (default: 4)\n");
    fprintf(stderr, "\nSpeculative decoding:\n");
    fprintf(stderr, "  --draft <f>    Draft model GGUF file (must share vocabulary)\n");
    fprintf(stderr, "  -d <int>       Draft tokens per step (default: 4)\n");
    fprintf(stderr, "\nAdvanced options:\n");
    fprintf(stderr, "  --json         Grammar-constrained JSON output mode\n");
    fprintf(stderr, "  --cache <file> KV cache file (saves/loads prompt state)\n");
}

static char *read_stdin(void) {
    size_t cap = 4096, len = 0;
    char *buf = (char *)malloc(cap);
    if (!buf) return NULL;

    int ch;
    while ((ch = fgetc(stdin)) != EOF) {
        if (len + 1 >= cap) {
            cap *= 2;
            char *tmp = (char *)realloc(buf, cap);
            if (!tmp) { free(buf); return NULL; }
            buf = tmp;
        }
        buf[len++] = (char)ch;
    }
    buf[len] = '\0';
    return buf;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }

    const char *model_path = argv[1];
    const char *prompt = NULL;
    int    max_tokens = 256;
    float  temperature = 0.8f;
    float  top_p = 0.9f;
    uint64_t seed = 42;
    int    context_override = 0;
    int    num_threads = 4;
    int    json_mode = 0;
    const char *cache_file = NULL;
    const char *draft_path = NULL;
    int    spec_k = 4;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            max_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            temperature = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "-k") == 0 && i + 1 < argc) {
            top_p = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            seed = (uint64_t)atoll(argv[++i]);
        } else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
            context_override = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-j") == 0 && i + 1 < argc) {
            num_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--json") == 0) {
            json_mode = 1;
        } else if (strcmp(argv[i], "--cache") == 0 && i + 1 < argc) {
            cache_file = argv[++i];
        } else if (strcmp(argv[i], "--draft") == 0 && i + 1 < argc) {
            draft_path = argv[++i];
        } else if (strcmp(argv[i], "-d") == 0 && i + 1 < argc) {
            spec_k = atoi(argv[++i]);
            if (spec_k < 1) spec_k = 1;
            if (spec_k > 16) spec_k = 16;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    char *stdin_prompt = NULL;
    if (!prompt) {
#ifdef _WIN32
        HANDLE h = GetStdHandle(STD_INPUT_HANDLE);
        DWORD mode;
        if (!GetConsoleMode(h, &mode)) {
            stdin_prompt = read_stdin();
            prompt = stdin_prompt;
        }
#else
        if (!isatty(fileno(stdin))) {
            stdin_prompt = read_stdin();
            prompt = stdin_prompt;
        }
#endif
    }

    if (!prompt || !*prompt) {
        fprintf(stderr, "No prompt provided. Use -p or pipe via stdin.\n");
        usage(argv[0]);
        return 1;
    }

    fprintf(stderr, "Loading model: %s\n", model_path);
    fprintf(stderr, "SIMD: %s\n",
#if defined(PICOLM_AVX2)
        "AVX2"
#elif defined(PICOLM_AVX)
        "AVX"
#elif defined(PICOLM_SSE3)
        "SSE3"
#elif defined(PICOLM_SSE2)
        "SSE2"
#elif defined(PICOLM_NEON)
        "NEON"
#else
        "scalar"
#endif
    );
    model_t model;
    if (model_load(&model, model_path, context_override) != 0) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    tensor_set_threads(num_threads);

    tokenizer_t tokenizer;
    if (tokenizer_load(&tokenizer, &model) != 0) {
        fprintf(stderr, "Failed to load tokenizer\n");
        model_free(&model);
        return 1;
    }

    sampler_t sampler;
    sampler_init(&sampler, temperature, top_p, seed);

    grammar_state_t grammar;
    grammar_init(&grammar, json_mode ? GRAMMAR_JSON : GRAMMAR_NONE, &tokenizer);
    if (json_mode)
        fprintf(stderr, "JSON grammar mode enabled\n");

    /* Load draft model if requested */
    model_t draft_model;
    memset(&draft_model, 0, sizeof(draft_model));
    int use_spec = 0;
    if (draft_path) {
        fprintf(stderr, "Loading draft model: %s\n", draft_path);
        if (model_load(&draft_model, draft_path, context_override) != 0) {
            fprintf(stderr, "Failed to load draft model — falling back to standard decoding\n");
        } else if (draft_model.config.vocab_size != model.config.vocab_size) {
            fprintf(stderr, "Draft/target vocab mismatch (%d vs %d) — disabling speculative decoding\n",
                    draft_model.config.vocab_size, model.config.vocab_size);
            model_free(&draft_model);
        } else if (json_mode) {
            fprintf(stderr, "Grammar mode + draft model: disabling speculative decoding\n");
            model_free(&draft_model);
        } else {
            use_spec = 1;
            fprintf(stderr, "Speculative decoding: %d draft tokens per step\n", spec_k);
        }
    }

    int vocab_size = model.config.vocab_size;

    /* Speculative decoding buffers: K+1 softmaxed target prob arrays + K draft token ids */
    float *tgt_probs = NULL;
    int   *draft_toks = NULL;
    if (use_spec) {
        tgt_probs  = (float *)malloc((size_t)(spec_k + 1) * vocab_size * sizeof(float));
        draft_toks = (int   *)malloc((size_t)spec_k * sizeof(int));
        if (!tgt_probs || !draft_toks) {
            fprintf(stderr, "OOM for speculative buffers — falling back to standard decoding\n");
            free(tgt_probs); free(draft_toks);
            tgt_probs = NULL; draft_toks = NULL;
            model_free(&draft_model);
            use_spec = 0;
        }
    }

    /* KV cache (target only) */
    int cache_pos = 0;
    if (cache_file)
        cache_pos = kvcache_load(&model, cache_file);

    /* Encode prompt */
    int max_prompt_tokens = (int)strlen(prompt) + 3;
    int *prompt_tokens = (int *)malloc((size_t)max_prompt_tokens * sizeof(int));
    int n_prompt = tokenizer_encode(&tokenizer, prompt, prompt_tokens, max_prompt_tokens, 1);

    int start_pos = 0;
    if (cache_pos > 0 && cache_pos <= n_prompt) {
        start_pos = cache_pos;
        fprintf(stderr, "Skipping %d cached prompt tokens\n", start_pos);
    }

    fprintf(stderr, "Prompt: %d tokens, generating up to %d (temp=%.2f, top_p=%.2f, threads=%d)\n",
            n_prompt, max_tokens, temperature, top_p, num_threads);
    fprintf(stderr, "---\n");

    int total_gen = 0;
    double t_start = get_time_ms();
    double t_first_token = 0;

    if (!use_spec) {
        /* ================================================================
         * Standard autoregressive generation (original loop)
         * ================================================================ */
        int token = prompt_tokens[start_pos > 0 ? start_pos - 1 : 0];
        int pos   = start_pos > 0 ? start_pos - 1 : 0;
        int total_steps = n_prompt + max_tokens;
        if (total_steps > model.config.max_seq_len)
            total_steps = model.config.max_seq_len;

        for (; pos < total_steps; pos++) {
            if (pos < start_pos) {
                token = prompt_tokens[pos];
                continue;
            }

            float *logits = model_forward(&model, token, pos);

            int next;
            if (pos < n_prompt - 1) {
                next = prompt_tokens[pos + 1];
            } else {
                if (pos == n_prompt - 1)
                    t_first_token = get_time_ms();

                grammar_apply(&grammar, logits, vocab_size);
                next = sampler_sample(&sampler, logits, vocab_size);
                grammar_advance(&grammar, &tokenizer, next);

                const char *piece = tokenizer_decode(&tokenizer, token, next);
                printf("%s", piece);
                fflush(stdout);

                total_gen++;

                if (next == (int)tokenizer.eos_id) break;
                if (grammar_is_complete(&grammar)) break;
            }

            token = next;
        }
    } else {
        /* ================================================================
         * Speculative decoding:
         *   1. Prefill both target and draft with all prompt tokens.
         *   2. Sample first generated token from target logits.
         *   3. Loop:
         *        a. Draft generates K tokens greedily (argmax).
         *        b. Target verifies K+1 positions (K drafts + bonus).
         *        c. Accept each draft token with probability
         *             min(1, p_target(x) / p_draft(x))          [softmax q]
         *        d. On rejection, resample from corrected dist
         *             max(0, p_target - p_draft), renormalized.
         *        e. If all K accepted, sample bonus from target_probs[K].
         * ================================================================ */

        /* Prefill phase: target from start_pos, draft always from 0 */
        t_first_token = get_time_ms(); /* will be overwritten at end of prefill */
        for (int pos = 0; pos < n_prompt - 1; pos++) {
            if (pos >= start_pos)
                model_forward(&model, prompt_tokens[pos], pos);
            model_forward(&draft_model, prompt_tokens[pos], pos);
        }

        /* Last prefill step: sample first generated token */
        t_first_token = get_time_ms();
        float *first_logits = model_forward(&model, prompt_tokens[n_prompt - 1], n_prompt - 1);
        model_forward(&draft_model, prompt_tokens[n_prompt - 1], n_prompt - 1);

        int cur_token = sampler_sample(&sampler, first_logits, vocab_size);
        {
            const char *piece = tokenizer_decode(&tokenizer, prompt_tokens[n_prompt - 1], cur_token);
            printf("%s", piece);
            fflush(stdout);
            grammar_advance(&grammar, &tokenizer, cur_token);
            total_gen++;
        }

        int cur_pos = n_prompt; /* next KV slot to fill */
        int spec_drafted = 0, spec_accepted = 0;

        if (cur_token == (int)tokenizer.eos_id || grammar_is_complete(&grammar))
            goto spec_done;

        while (total_gen < max_tokens && cur_pos < model.config.max_seq_len) {
            /* How many draft tokens to attempt this round */
            int max_n = spec_k;
            if (cur_pos + max_n + 1 > model.config.max_seq_len)
                max_n = model.config.max_seq_len - cur_pos - 1;
            if (max_tokens - total_gen - 1 < max_n)
                max_n = max_tokens - total_gen - 1;
            if (max_n < 0) max_n = 0;

            /* ---- Draft: greedy argmax for max_n steps ---- */
            int n = 0;
            int d_cur = cur_token;
            for (n = 0; n < max_n; n++) {
                float *dl = model_forward(&draft_model, d_cur, cur_pos + n);
                /* greedy pick from raw logits */
                int best = 0;
                for (int v = 1; v < vocab_size; v++)
                    if (dl[v] > dl[best]) best = v;
                draft_toks[n] = best;
                d_cur = best;
                if (best == (int)tokenizer.eos_id) { n++; break; }
            }
            spec_drafted += n;

            /* ---- Target: verify n positions + 1 bonus ---- */
            int t_cur = cur_token;
            for (int k = 0; k <= n; k++) {
                float *tl = model_forward(&model, t_cur, cur_pos + k);
                float *tp = tgt_probs + (size_t)k * vocab_size;
                memcpy(tp, tl, (size_t)vocab_size * sizeof(float));
                /* apply temperature then softmax to get true target distribution */
                if (sampler.temperature > 0.0f) {
                    float inv_t = 1.0f / sampler.temperature;
                    for (int v = 0; v < vocab_size; v++) tp[v] *= inv_t;
                }
                softmax(tp, vocab_size);
                if (k < n) t_cur = draft_toks[k];
            }

            /* ---- Accept / reject ---- */
            int accepted = 0;
            int rejected = 0; /* set to 1 when we resample and break */

            for (int k = 0; k < n; k++) {
                int dt = draft_toks[k];
                float *tp = tgt_probs + (size_t)k * vocab_size;

                /* draft q: softmax of draft logits at position k.
                 * We ran the draft with argmax but stored the full distribution.
                 * Re-run draft to get q? No — we only kept the chosen token.
                 * Treat draft as one-hot: accept with prob min(1, p_target(dt)). */
                float p = tp[dt];
                float r = sampler_rand(&sampler);

                if (r < p) {
                    /* Accept draft token */
                    const char *piece = tokenizer_decode(&tokenizer, cur_token, dt);
                    printf("%s", piece);
                    fflush(stdout);
                    grammar_advance(&grammar, &tokenizer, dt);
                    total_gen++;
                    accepted++;
                    cur_token = dt;
                    if (dt == (int)tokenizer.eos_id || grammar_is_complete(&grammar)) {
                        spec_accepted += accepted;
                        cur_pos += accepted;
                        goto spec_done;
                    }
                } else {
                    /* Reject: resample from corrected dist = p(x) with dt zeroed out */
                    tp[dt] = 0.0f;
                    float sum = 0.0f;
                    for (int v = 0; v < vocab_size; v++) sum += tp[v];

                    int next;
                    if (sum < 1e-10f) {
                        /* fallback: argmax of original target probs */
                        float *tp0 = tgt_probs + (size_t)k * vocab_size;
                        next = 0;
                        for (int v = 1; v < vocab_size; v++)
                            if (tp0[v] > tp0[next]) next = v;
                    } else {
                        float rr = sampler_rand(&sampler) * sum;
                        float acc = 0.0f;
                        next = 0;
                        for (int v = 0; v < vocab_size; v++) {
                            acc += tp[v];
                            if (acc >= rr) { next = v; break; }
                        }
                    }

                    const char *piece = tokenizer_decode(&tokenizer, cur_token, next);
                    printf("%s", piece);
                    fflush(stdout);
                    grammar_advance(&grammar, &tokenizer, next);
                    total_gen++;
                    spec_accepted += accepted;
                    cur_token = next;
                    cur_pos += accepted + 1;
                    rejected = 1;

                    if (next == (int)tokenizer.eos_id || grammar_is_complete(&grammar))
                        goto spec_done;
                    break;
                }
            }

            if (!rejected) {
                /* All n draft tokens accepted: sample bonus from target at position n */
                float *tp = tgt_probs + (size_t)n * vocab_size;
                int bonus = sampler_sample_probs(&sampler, tp, vocab_size);
                const char *piece = tokenizer_decode(&tokenizer, cur_token, bonus);
                printf("%s", piece);
                fflush(stdout);
                grammar_advance(&grammar, &tokenizer, bonus);
                total_gen++;
                spec_accepted += n;
                cur_token = bonus;
                cur_pos += n + 1;

                if (bonus == (int)tokenizer.eos_id || grammar_is_complete(&grammar))
                    goto spec_done;
            }
        }

spec_done:;
        fprintf(stderr, "Speculative: drafted=%d accepted=%d (%.1f%%)\n",
                spec_drafted, spec_accepted,
                spec_drafted > 0 ? 100.0f * spec_accepted / spec_drafted : 0.0f);
    }

    printf("\n");
    double t_end = get_time_ms();

    if (cache_file && n_prompt > 0)
        kvcache_save(&model, cache_file, n_prompt);

    double total_time  = (t_end - t_start) / 1000.0;
    if (t_first_token == 0) t_first_token = t_end;
    double gen_time    = (t_end - t_first_token) / 1000.0;
    double prefill_time = (t_first_token - t_start) / 1000.0;
    int actual_prefill = n_prompt - start_pos;
    if (actual_prefill < 0) actual_prefill = 0;

    fprintf(stderr, "---\n");
    fprintf(stderr, "Prefill: %d tokens in %.2fs (%.1f tok/s)%s\n",
            actual_prefill, prefill_time,
            prefill_time > 0 ? (double)actual_prefill / prefill_time : 0,
            start_pos > 0 ? " [partially cached]" : "");
    fprintf(stderr, "Generation: %d tokens in %.2fs (%.1f tok/s)\n",
            total_gen, gen_time,
            gen_time > 0 ? (double)total_gen / gen_time : 0);
    fprintf(stderr, "Total: %.2fs\n", total_time);
    fprintf(stderr, "Memory: %.2f MB runtime state (FP16 KV cache)\n",
            (double)model.state.mem_size / (1024.0 * 1024.0));

    free(tgt_probs);
    free(draft_toks);
    grammar_free(&grammar);
    free(prompt_tokens);
    free(stdin_prompt);
    tokenizer_free(&tokenizer);
    if (use_spec) model_free(&draft_model);
    model_free(&model);

    return 0;
}
