/**
 * @file src/qwen.c
 */

#include "qwen.h"
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

/**
 * Transformer
 * @{
 */
Qwen* qwen_create(QwenConfig* cfg) {
    Qwen* q = calloc(1, sizeof(Qwen));
    if (!q) {
        return NULL;
    }

    // build the Tokenizer via the tokenizer .bin file
    q->tokenizer = tokenizer_create(cfg->path);
    if (!q->tokenizer) {
        goto tokenizer_failed;
    }

    // build the Transformer via the model .bin file
    q->model = model_create(cfg->path, cfg->seq_len);
    if (!q->model) {
        goto model_failed;
    }

    // build the Sampler
    q->sampler = sampler_create(q->tokenizer->vocab_size, cfg->temperature, cfg->top_p, cfg->seed);
    if (!q->sampler) {
        goto sampler_failed;
    }

    q->config = cfg;

    return q;

sampler_failed:
    model_free(q->model);
model_failed:
    tokenizer_free(q->tokenizer);
tokenizer_failed:
    free(q);
    return NULL;
}

void qwen_free(Qwen* q) {
    if (q) {
        if (q->model) {
            model_free(q->model);
        }
        if (q->tokenizer) {
            tokenizer_free(q->tokenizer);
        }
        if (q->sampler) {
            sampler_free(q->sampler);
        }
        free(q);
    }
}

/** @} */
