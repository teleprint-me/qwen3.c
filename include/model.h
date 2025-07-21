/**
 * @file include/model.h
 *
 * Qwen3 has the following features:
 *   - Type: Causal Language Models
 *   - Training Stage: Pretraining & Post-training
 *   - Number of Parameters: 0.6B, 1.7B, and 4B
 *   - Number of Embedding Parameters: ~0.4B
 *   - Number of Layers: 0.6B/1.7B -> 28, 4B -> 36
 *   - Number of Attention Heads (GQA): 0.6B/1.7B -> 16 for Q, 4B -> 32 for Q,
 * always 8 for KV
 *   - Context Length: 32768 natively and 131072 tokens with YaRN.
 *
 * @ref https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f
 */

#ifndef QWEN_MODEL_H
#define QWEN_MODEL_H

#include "q8.h"
#include <sys/types.h>

#define QWEN_MAGIC 0x7177656E
#define QWEN_VERSION 1

/**
 * CONFIGURATION
 */

typedef struct ModelParams {
    int magic; // checkpoint magic number
    int version; // file format version
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
    int head_dim; // head dimension
    int shared_classifier; // 1 if cls == p_tokens
    int block_size; // quantization block size (weights.py uses 64)
} ModelParams;

/**
 * WEIGHTS
 */

/**
 * Note: weights.* fields are either:
 *   - Arrays of Q8Tensors, one per layer
 *   - Flat fp32 arrays (e.g. for norms)
 *   - Formatted as weights.wq[layer] → Q8Tensor {q, s}
 */
typedef struct ModelWeights {
    // Attention weights
    Q8Tensor* wq; // (n_layers, dim, n_heads * head_dim)
    Q8Tensor* wk; // (n_layers, dim, n_kv_heads * head_dim)
    Q8Tensor* wv; // (n_layers, dim, n_kv_heads * head_dim)
    Q8Tensor* wo; // (n_layers, n_heads * head_dim, dim)

    // Feed-forward network
    Q8Tensor* w1; // (n_layers, hidden_dim, dim)
    Q8Tensor* w2; // (n_layers, dim, hidden_dim)
    Q8Tensor* w3; // (n_layers, hidden_dim, dim)

    // (optional) classifier weights for the logits, on the last layer
    Q8Tensor* cls;

    // Token embedding
    Q8Tensor* qe; // quantized embedding (vocab_size, dim)
    float* fe; // dequantized token embeddings (vocab_size, dim)

    // RMSNorm weights
    float* att_rms_norm; // (n_layers, dim)
    float* ffn_rms_norm; // (n_layers, dim)
    float* out_rms_norm; // (dim)

    // QK-RMSNorm for Qwen3
    float* q_rms_norm;
    float* k_rms_norm;
} ModelWeights;

/**
 * STATE
 */

/**
 * Scratch space reused across the forward pass.
 * Buffers are overwritten at each layer.
 */
typedef struct ForwardState {
    // Residual stream
    float* x; // Persistent residual (dim)
    float* x_rms_norm; // RMSNorm(x) (n_heads * head_dim)

    // Attention workspace
    float* q; // Query (n_heads * head_dim)
    float* k; // Key   (n_kv_heads * head_dim)
    float* v; // Value (n_kv_heads * head_dim)
    float* scores; // Attention scores (n_heads * seq_len)

    // Feed-forward network
    float* mlp_in; // w1(x) = mlp_in (hidden_dim)
    float* mlp_gate; // w3(x) = mlp_gate (hidden_dim)

    // Output
    float* logits; // Final output logits (vocab_size)

    // Key/value cache
    float* k_cache; // Cached keys (n_layers, seq_len, kv_dim)
    float* v_cache; // Cached values (n_layers, seq_len, kv_dim)

    // Quantized buffers
    Q8Tensor qx; // Quantized input to attention (dim)
    Q8Tensor qh; // Quantized input to FFN (hidden_dim)
} ForwardState;

/**
 * TRANSFORMER
 */

typedef struct Model {
    ModelParams params; // model architecture + hyperparameters
    ModelWeights weights; // model weights (quantized + fp32 norms)
    ForwardState state; // forward pass scratch space
    void* data; // read-only pointer to memory-mapped model file
    ssize_t size; // size of the memory-mapped model file
} Model;

/**
 * @brief Construct a Transformer model from a memory-mapped checkpoint file.
 *
 * This function encapsulates the entire setup process:
 * - Maps the checkpoint file into memory via `mmap`
 * - Parses and validates the `Params` header
 * - Loads FP32 weights and allocates Q8 quantized tensors
 * - Allocates and initializes model state buffers
 *
 * The returned pointer owns all necessary memory, including:
 * - Memory-mapped file region (`t->model`)
 * - Quantized weight tensors
 * - Dequantized embeddings
 * - Transformer internal state buffers
 *
 * The pointer must be released using `transformer_free()` to avoid leaks.
 *
 * @param path              Path to the checkpoint file on disk.
 * @param override_seq_len  Optional context length override (0 to use
 * checkpoint default).
 * @return Pointer to a fully constructed Transformer, or NULL on failure.
 */
Model* model_create(const char* path, int override_seq_len);

/**
 * @brief Frees all memory associated with a Transformer model.
 *
 * This includes:
 * - Quantized and dequantized weight buffers
 * - Attention, MLP, and cache buffers in the model state
 * - The memory-mapped checkpoint file
 * - The Transformer struct itself
 *
 * Safe to call on NULL.
 *
 * @param m Pointer to a Transformer created by `transformer_create`.
 */
void model_free(Model* m);

#endif // QWEN_MODEL_H
