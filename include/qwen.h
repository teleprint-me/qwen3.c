/**
 * @file qwen.h
 * @brief High-level API for initializing and managing the Qwen3 Transformer
 * model.
 *
 * This header provides a compact interface for creating a Qwen inference
 * engine, including its core components:
 *  - Memory-mapped model weights (quantized int8 + FP32 norms)
 *  - Tokenizer for UTF-8/BPE text encoding and decoding
 *  - Sampler for autoregressive token generation (temperature, top-p)
 *
 * The Qwen engine is configured at initialization via a user-managed
 * QwenConfig, which specifies runtime parameters such as sequence length,
 * temperature, and random seed. Once created, the Qwen object can be passed
 * to completion and chat_completion functions for text generation.
 *
 * @note This API is specific to Qwen models and assumes a compatible
 * binary format exported via `weights.py` and `tokenizer.py`.
 */

#ifndef QWEN_H
#define QWEN_H

#include "tokenizer.h"
#include "model.h"
#include "sampler.h"
#include <stdint.h>

/**
 * @enum Think
 * @brief Whether the model should include its "reasoning" tokens (<think>).
 *
 * Reasoning mode injects special thinking tokens into chat completions,
 * allowing the model to produce chain-of-thought style responses.
 * This has no effect on simple completions.
 */
typedef enum Think {
    THINK_OFF,  ///< Omit reasoning tokens (<think>...</think>)
    THINK_ON  ///< Include reasoning tokens
} Think;

/**
 * @struct QwenConfig
 * @brief Configuration options for Qwen initialization.
 *
 * The config is user-managed and passed to `qwen_create()`.
 * It is not owned or modified by the Qwen engine, so it may be stack-
 * or statically-allocated by the caller.
 *
 * Example:
 * @code
 * QwenConfig cfg = {
 *     .path = "model/Qwen3-1.7B-Q8.bin",
 *     .think = THINK_ON,
 *     .seed = 1337,
 *     .temperature = 1.0f,
 *     .top_p = 0.9f,
 *     .seq_len = 32768
 * };
 * Qwen* q = qwen_create(&cfg);
 * completion(q, "Hello!");
 * qwen_free(q);
 * @endcode
 */
typedef struct QwenConfig {
    const char* path;  ///< Path to the Qwen model binary (used for both weights and tokenizer)
    Think think;  ///< Reasoning mode (only affects chat completions)
    uint64_t seed;  ///< Random seed for sampling (fixed for reproducibility)
    float temperature;  ///< Softmax temperature (>0.0; lower = deterministic, higher = creative)
    float top_p;  ///< Nucleus sampling probability mass (1.0 = disabled)
    int seq_len;  ///< Maximum sequence length (context window override)
} QwenConfig;

/**
 * @struct Qwen
 * @brief Fully initialized Qwen3 inference engine.
 *
 * The Qwen object encapsulates all resources required for autoregressive
 * text generation, including:
 *  - Memory-mapped model weights (`model`)
 *  - Tokenizer vocabulary and BPE merge table (`tokenizer`)
 *  - Sampler for stochastic decoding (`sampler`)
 *
 * The `config` pointer references the user-supplied configuration.
 * The config must remain valid for the lifetime of the Qwen instance.
 */
typedef struct Qwen {
    Model* model;  ///< Transformer weights and forward state
    Tokenizer* tokenizer;  ///< Tokenizer for text encoding/decoding
    Sampler* sampler;  ///< Sampler for stochastic decoding
    const QwenConfig* config;  ///< User-provided configuration (not owned)
} Qwen;

/**
 * @brief Creates and initializes a Qwen inference engine.
 *
 * This function:
 *  - Loads the tokenizer (`.tokenizer` binary, derived from `path`)
 *  - Memory-maps the quantized model weights
 *  - Allocates a sampler with temperature and top-p parameters
 *
 * On success, returns a pointer to a fully constructed Qwen engine.
 * On failure, returns NULL and releases all partially allocated resources.
 *
 * @param cfg Pointer to a user-managed QwenConfig (must outlive the Qwen instance).
 * @return Qwen* Initialized inference engine, or NULL on failure.
 */
Qwen* qwen_create(QwenConfig* cfg);

/**
 * @brief Frees all resources owned by a Qwen inference engine.
 *
 * This releases:
 *  - Tokenizer tables
 *  - Model weights and forward state
 *  - Sampler buffers
 *
 * The config referenced by `qwen->config` is **not** freed, as it is
 * user-managed.
 *
 * Safe to call on NULL.
 *
 * @param q Pointer to a Qwen instance created by `qwen_create()`.
 */
void qwen_free(Qwen* q);

#endif  // QWEN_H
