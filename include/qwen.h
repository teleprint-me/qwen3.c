/**
 * @file include/qwen.h
 */

#ifndef QWEN_H
#define QWEN_H

#include "tokenizer.h"
#include "model.h"
#include "sampler.h"
#include <stdint.h>

typedef enum Think {
    THINK_OFF,
    THINK_ON,
} Think;

typedef struct QwenConfig {
    const char* path;
    Think think; // exclusive to chat completion
    uint64_t seed;
    float temperature;
    float top_p;
    int seq_len; // override max sequence length
} QwenConfig;

typedef struct Qwen {
    Model* model;
    Tokenizer* tokenizer;
    Sampler* sampler;
    QwenConfig* config;
} Qwen;

Qwen* qwen_create(QwenConfig* cfg);
void qwen_free(Qwen* q);

#endif // QWEN_H
