#ifndef SAMPLER_H
#define SAMPLER_H

#include <stdint.h>

typedef struct {
    float    temperature;
    float    top_p;
    uint64_t rng_state;   /* xorshift64 state */
} sampler_t;

/* Initialize sampler with given parameters */
void sampler_init(sampler_t *s, float temperature, float top_p, uint64_t seed);

/* Sample a token index from logits[vocab_size].
 * Modifies logits in-place (temperature scaling, softmax). */
int sampler_sample(sampler_t *s, float *logits, int vocab_size);

/* Sample from a pre-computed probability distribution (already softmaxed).
 * Applies top-p but not temperature (probs already scaled). */
int sampler_sample_probs(sampler_t *s, float *probs, int vocab_size);

/* Return a uniform float in [0, 1) using the sampler's RNG. */
float sampler_rand(sampler_t *s);

#endif /* SAMPLER_H */
