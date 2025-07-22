/**
 * @file include/sampler.h
 * @brief The Sampler, which takes logits and returns a sampled token
 * sampling can be done in a few ways: greedy argmax, sampling, top-p sampling
 */

#ifndef QWEN_SAMPLER_H
#define QWEN_SAMPLER_H

#include <stdint.h>

/**
 * @brief Probability-Index pair for top-p filtering and sampling.
 */
typedef struct Probability {
    float sample; // probability mass
    int index; // token index
} Probability;

typedef struct Sampler {
    Probability* dist; // buffer used in top-p sampling
    uint64_t seed; // current rng state
    float temperature; // entropy scale
    float top_p; // top probability
    int vocab_size; // embed dim
} Sampler;

Sampler* sampler_create(int vocab_size, float temperature, float top_p, uint64_t seed);
void sampler_free(Sampler* s);

/**
 * @brief Draws the next token from the model's output distribution.
 * Internally applies temperature scaling, softmax, and nucleus sampling.
 *
 * @param s Sampler state (holds RNG, buffers, etc.)
 * @param logits Model output logits (modified in-place by softmax)
 * @return Sampled token ID
 */
int sample(Sampler* s, float* logits);

#endif // QWEN_SAMPLER_H
