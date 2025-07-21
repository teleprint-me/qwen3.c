/**
 * @file src/sampler.c
 * @brief The Sampler, which takes logits and returns a sampled token
 * sampling can be done in a few ways: greedy argmax, sampling, top-p sampling
 */

#include "xorshift.h"
#include "forward.h"
#include "sampler.h"
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

/**
 * @section Public Sampler Lifecycle
 * @{
 */

Sampler* sampler_create(
    int vocab_size, float temperature, float top_p, uint64_t seed
) {
    Sampler* s = calloc(1, sizeof(Sampler));
    if (!s) {
        return NULL;
    }

    // buffer only used with nucleus sampling; may not need but it's ~small
    s->dist = calloc(vocab_size, sizeof(Probability));
    if (!s->dist) {
        free(s);
        return NULL;
    }

    // clamp to [ε, 1.0f], where ε > 0
    const float epsilon = 1e-6f;
    if (top_p > 1.0f || isnan(top_p) || 1 == isinf(top_p)) {
        s->top_p = 1.0f;  // upper limit
    } else if (top_p < epsilon || -1 == isinf(top_p)) {
        s->top_p = epsilon;  // mitigate low probs
    } else {
        s->top_p = top_p;
    }

    // clamp to [ε, +/- 1.0f], where ε > 0
    if (isnan(temperature) || 1 == isinf(temperature)) {
        s->temperature = 1.0f;  // sane default
    } else if (temperature < epsilon || -1 == isinf(temperature)) {
        s->temperature = epsilon;  // mitigate division by zero
    } else {
        s->temperature = temperature;
    }

    s->vocab_size = vocab_size;
    s->seed = seed;

    fprintf(stderr, "[Sampler] vocab_size=%d\n", s->vocab_size);
    fprintf(stderr, "[Sampler] temperature=%f\n", (double) s->temperature);
    fprintf(stderr, "[Sampler] top_p=%f\n", (double) s->top_p);
    fprintf(stderr, "[Sampler] seed=%lu\n", s->seed);

    return s;
}

void sampler_free(Sampler* s) {
    if (s) {
        if (s->dist) {
            free(s->dist);
        }
        free(s);
    }
}

/** @} */

/**
 * @section Private sampling utilities
 * @{
 */

/**
 * @brief Given a sorted Probability array, find smallest prefix
 * such that cumulative sum exceeds `top_p`.
 *
 * @param out_mass Optional: returns cumulative mass of included tokens
 * @return Index of last included token
 */
int sampler_mass_index(Sampler* s, float* out_mass) {
    float mass = 0.0f;
    int id = s->vocab_size - 1;

    for (int i = 0; i < s->vocab_size; i++) {
        mass += s->dist[i].sample;
        if (mass > s->top_p) {
            id = i;
            break;
        }
    }

    if (out_mass) {
        // Heal the sampled distribution
        const float epsilon = 1e-3f;  // absolute threshold
        if (mass < epsilon) {
            for (int i = 0; i <= id; i++) {
                mass += s->dist[i].sample;  // add weight to the distribution
            }
        }
        *out_mass = mass;
    }

    return id;
}

/**
 * @brief Sample index from a truncated distribution using inverse CDF.
 *
 * @ref https://www.probabilitycourse.com/chapter3/3_2_1_cdf.php
 *
 * @param dist Array of Probability structs
 * @param n Number of valid entries
 * @param coin Random number in [0,1)
 * @param total_mass Sum of the truncated distribution
 * @return Sampled index from dist
 */
int sampler_cdf_index(Probability* dist, int n, float coin, float total_mass) {
    float cdf = 0.0f;
    float r = coin * total_mass;
    for (int i = 0; i <= n; i++) {
        cdf += dist[i].sample;
        if (r < cdf) {
            return dist[i].index;
        }
    }
    return dist[n - 1].index;  // fallback
}

/**
 * @brief Comparator for qsort: descending order by probability
 * @ref https://en.cppreference.com/w/c/algorithm/qsort.html
 */
int sampler_cmp_dist(const void* a, const void* b) {
    const Probability* n = (const Probability*) a;
    const Probability* m = (const Probability*) b;
    if (n->sample > m->sample) {
        return -1;
    }
    if (n->sample < m->sample) {
        return 1;
    }
    return 0;
}

/**
 * @brief Nucleus Sampling (Top-p Sampling).
 *
 * Truncates the distribution to the smallest prefix whose cumulative
 * probability exceeds `top_p`, then samples from this renormalized subset.
 *
 * @ref Holtzman et al., 2020 (https://arxiv.org/abs/1904.09751)
 *
 * @param samples Input probability distribution (softmaxed logits)
 * @param coin Random float in [0, 1)
 * @return Sampled token index
 */
int sampler_top_p(Sampler* s, float* samples, float coin) {
    // Build full probability-index mapping
    for (int i = 0; i < s->vocab_size; i++) {
        s->dist[i].index = i;
        s->dist[i].sample = samples[i];
    }

    // Sort descending by probability
    qsort(s->dist, s->vocab_size, sizeof(Probability), sampler_cmp_dist);

    // Truncate where cumulative probability exceeds top_p
    float mass = 0.0f;
    int id = sampler_mass_index(s, &mass);
    // Sample from truncated distribution
    return sampler_cdf_index(s->dist, id, coin, mass);
}

/** @} */

/**
 * @section Public sampler
 * @{
 */

int sample(Sampler* s, float* logits) {
    // apply the temperature to the logits
    for (int q = 0; q < s->vocab_size; q++) {
        logits[q] /= s->temperature;  // scale
    }

    // apply softmax to the logits to get the samples for next token
    softmax(logits, s->vocab_size);  // normalize
    // create a source of entropy for sampling
    float coin = xorshift_float(&s->seed);  // flip a coin
    // top-p (nucleus) sampling, clamping the least likely tokens to zero
    return sampler_top_p(s, logits, coin);
}

/** @} */
