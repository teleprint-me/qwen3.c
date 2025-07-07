/**
 * @file include/model.h
 *
 * Qwen3 has the following features:
 *   - Type: Causal Language Models
 *   - Training Stage: Pretraining & Post-training
 *   - Number of Parameters: 0.6B, 1.7B, and 4B
 *   - Number of Embedding Parameters: ~0.4B
 *   - Number of Layers: 0.6B/1.7B -> 28, 4B -> 36
 *   - Number of Attention Heads (GQA): 0.6B/1.7B -> 16 for Q, 4B -> 32 for Q, always 8 for KV
 *   - Context Length: 32768 natively and 131072 tokens with YaRN.
 *
 * @ref https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f
 */

#ifndef QWEN_MODEL_H
#define QWEN_MODEL_H

#include "q8.h"
#include <sys/types.h>

/**
 * CONFIGURATION
 */

typedef struct Params {
    int magic_number; // checkpoint magic number ("qwen")
    int version; // file format version
    int dim; // transformer width (e.g. 2048)
    int hidden_dim; // FFN inner dimension (e.g. 6144)
    int n_layers; // number of transformer layers
    int n_heads; // number of attention heads
    int n_kv_heads; // number of key/value heads (multiquery if < n_heads)
    int vocab_size; // number of tokens in vocabulary
    int seq_len; // maximum sequence length (e.g. 40960)
    int head_dim; // dimension per head
    int shared_classifier; // if true, cls = token embedding weights
    int group_size; // quantization group size (typically 64)
} Params;

/**
 * WEIGHTS
 */

/**
 * Note: weights.* fields are either:
 *   - Arrays of Q8Tensors, one per layer
 *   - Flat fp32 arrays (e.g. for norms)
 */
typedef struct Weights {
    // Attention weights
    Q8Tensor* wq; // (n_layers, dim, n_heads * head_dim)
    Q8Tensor* wk; // (n_layers, dim, n_kv_heads * head_dim)
    Q8Tensor* wv; // (n_layers, dim, n_kv_heads * head_dim)
    Q8Tensor* wo; // (n_layers, n_heads * head_dim, dim)

    // Feed-forward network
    Q8Tensor* w1; // (n_layers, hidden_dim, dim)
    Q8Tensor* w2; // (n_layers, dim, hidden_dim)
    Q8Tensor* w3; // (n_layers, hidden_dim, dim)

    // Output classifier (optional)
    Q8Tensor* cls; // (vocab_size, dim) or NULL if shared

    // Token embedding
    Q8Tensor* qe; // quantized embedding (vocab_size, dim)
    float* fe; // dequantized token embeddings (vocab_size, dim)

    // RMSNorm weights
    float* att_rms_norm; // (n_layers, dim)
    float* ffn_rms_norm; // (n_layers, dim)
    float* out_rms_norm; // (dim)

    // Qwen3 layernorms for Q and K
    float* q_rms_norm; // (n_layers, head_dim)
    float* k_rms_norm; // (n_layers, head_dim)
} Weights;

/**
 * STATE
 */

/**
 * Scratch space reused across the forward pass.
 * Buffers are overwritten at each layer.
 */
typedef struct State {
    // Residual stream
    float* x; // persistent residual (dim)
    float* r; // normalized / projected buffer (n_heads * head_dim)
    float* att_out; // attention output before residual (dim)

    // Attention workspace
    float* q; // query (n_heads * head_dim)
    float* k; // key   (n_kv_heads * head_dim)
    float* v; // value (n_kv_heads * head_dim)
    float* att; // attention scores (n_heads * seq_len)
    float* logits; // final output logits (vocab_size)

    // Key/value cache
    float* k_cache; // (n_layers, seq_len, kv_dim)
    float* v_cache; // (n_layers, seq_len, kv_dim)

    // FFN
    float* mlp_in; // result of w1(x) (hidden_dim)
    float* mlp_gate; // result of w3(x) (hidden_dim)
    Q8Tensor qx; // quantized residual input (dim)
    Q8Tensor qh; // quantized mlp_in (hidden_dim)
} State;

/**
 * TRANSFORMER
 */

typedef struct Transformer {
    float* data; // pointer to memory-mapped model weights
    Params params; // model architecture + hyperparameters
    Weights weights; // model weights (quantized + fp32 norms)
    State state; // forward pass scratch space
    ssize_t file_size;
} Transformer;

State* state_create(Params* p);
void state_free(State* s);

/**
 * @brief Initialize and allocate quantized and fp32 weight tensors from memory-mapped stream.
 *
 * This function assumes `stream` points to a contiguous memory-mapped model checkpoint.
 * All fp32 weights (e.g. RMSNorm parameters) are read first, then the quantized tensors
 * are constructed using `q8_tensor`, which allocates memory internally and adjusts the stream pointer.
 *
 * @param p        Pointer to model configuration parameters.
 * @param stream   Pointer to memory-mapped weights block.
 * @return Pointer to allocated Weights struct, or NULL on error.
 */
Weights* weights_create(Params* p, void* stream);
void weights_free(Params* p, Weights* w);

#endif // QWEN_MODEL_H
