/**
 * @file examples/qwen.c
 * @brief Inference for Qwen-3 Transformer model in pure C, int8 quantized forward pass.
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

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/mman.h>
#include <stdio.h>
#include <ctype.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include <omp.h>

#define QWEN_MAGIC 0x7177656E
#define QWEN_VERSION 1

#define QTKN_MAGIC 0x71746B6E
#define QTKN_VERSION 1

#define MAX_SEQ_LEN 32768

/**
 * @section 8-bit symmetric quantization utilities for Qwen model inference.
 *
 * @todo q8: Memory Management Refactor
 *   - Decouple Q8Tensor mapping (mmap) from allocation (malloc)
 *   - Use explicit ownership flags or wrapper structs if heap allocation is generalized
 *   - Replace GS global with model-specific group size (Params.group_size)
 *
 * @{
 */

/// @note Replace with `Params.group_size` during full allocator refactor.
/// Currently required to calculate values per quantization group.
int GS = 2; // global quantization group size

/**
 * @struct Q8Tensor
 * @brief A quantized tensor with int8 values and per-group scale factors.
 */
typedef struct Q8Tensor {
    float* s;     ///< scaling factors per group
    int8_t* q;    ///< quantized values
} Q8Tensor;

/**
 * @brief Quantize a float32 tensor symmetrically to int8 using Q8_0 format.
 *
 * @param qt Q8Tensor to write quantized output into.
 * @param x  Input float32 tensor of length `n`.
 * @param n  Total number of elements in the input/output.
 */
void q8_quantize(Q8Tensor* qt, float* x, int n) {
    const int num_groups = n / GS;
    const float Q_MAX = 127.0f;

    for (int group = 0; group < num_groups; group++) {
        float* xg  = x + group * GS;
        int8_t* qg = qt->q + group * GS;

        // Find max absolute value
        float wmax = fabsf(xg[0]);
        #pragma omp simd reduction(max:wmax)
        for (int i = 1; i < GS; i++) {
            wmax = fmaxf(wmax, fabsf(xg[i]));
        }

        // Compute scaling factor
        float scale = (wmax == 0.0f) ? 1e-6f : (wmax / Q_MAX);
        qt->s[group] = scale;

        // Quantize with clamping to [-127, 127]
        #pragma omp parallel for
        for (int i = 0; i < GS; i++) {
            float q = xg[i] / scale;
            qg[i] = (int8_t) fminf(fmaxf(roundf(q), -Q_MAX), Q_MAX);
        }
    }
}

/**
 * @brief Dequantize int8 tensor back to float32 using per-group scale factors.
 *
 * @param qt Q8Tensor to read from.
 * @param x  Output float32 tensor of length `n`.
 * @param n  Number of elements to dequantize.
 */
void q8_dequantize(Q8Tensor* qt, float* x, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        x[i] = qt->q[i] * qt->s[i / GS];
    }
}

/**
 * @brief Map an array of Q8Tensor structs to a linear buffer (e.g. from mmap).
 *
 * This function does not allocate memory for `q` or `s`. It only allocates
 * the outer array of Q8Tensor structs and sets internal pointers to offsets
 * within the given `buffer`.
 *
 * @param buffer Starting address of a memory-mapped region.
 * @param n      Number of Q8Tensor structs to map.
 * @param size   Number of quantized values per tensor.
 * @return       Pointer to allocated Q8Tensor array (free with `free()` only).
 *
 * @note Ownership:
 *   - Do NOT free `q` or `s` of the returned tensors.
 *   - Only the outer array returned by this function should be freed.
 */
Q8Tensor* q8_tensor_map(void* buffer, int n, int size) {
    if (!buffer || n <= 0 || size <= 0) return NULL;

    Q8Tensor* qt = calloc(n, sizeof(Q8Tensor));
    if (!qt) return NULL;

    for (int i = 0; i < n; i++) {
        /* map quantized int8 values*/
        qt[i].q = (int8_t*) buffer;
        buffer = (int8_t*) buffer + size;

        /* map scale factors */
        qt[i].s = (float*) buffer;
        buffer = (float*) buffer + size / GS;
    }

    return qt;
}

/** @} */

/**
 * @section Qwen3ForCausalLM Model Loader
 * @{
 */

typedef struct Params {
    int magic; // checkpoint magic number
    int version; // file format version
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
    int head_dim; // head dimension
    int shared_classifier; // 1 if cls == p_tokens
    int group_size; // quantization group size (export.py uses 64)
} Params;

/**
 * Note: weights.* fields are either:
 *   - Arrays of Q8Tensors, one per layer
 *   - Flat fp32 arrays (e.g. for norms)
 *   - Formatted as weights.wq[layer] → Q8Tensor {q, s}
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

    // (optional) classifier weights for the logits, on the last layer
    Q8Tensor *cls;

    // Token embedding
    Q8Tensor* qe; // quantized embedding (vocab_size, dim)
    float* fe; // dequantized token embeddings (vocab_size, dim)

    // RMSNorm weights
    float* att_rms_norm; // (n_layers, dim)
    float* ffn_rms_norm; // (n_layers, dim)
    float* out_rms_norm; // (dim)

    // QK-RMSNorm for Qwen3
    float *q_rms_norm;
    float *k_rms_norm;
} Weights;

typedef struct ForwardState {
    // Residual stream
    float* x; // Persistent residual (dim)
    float* x_rms_norm; // RMSNorm(x), reused in scores and ffn (n_heads * head_dim)
    
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

typedef struct Transformer {
    Params params; // model architecture + hyperparameters
    Weights weights; // model weights (quantized + fp32 norms)
    ForwardState state; // forward pass scratch space
    void* model; // read-only pointer to memory-mapped model file
    ssize_t size; // size of the memory-mapped model file
} Transformer;

/**
 * @section Model Checkpoint
 */

bool model_read_checkpoint(Transformer* t, const char* path) {
    FILE* file = fopen(path, "rb");
    if (!file) {
        goto open_failure;
    }

    if (-1 == fseek(file, 0, SEEK_END)) {
        goto read_failure;
    }

    t->size = ftell(file);
    if (-1 == t->size) {
        goto read_failure;
    }

    t->model = mmap(NULL, t->size, PROT_READ, MAP_PRIVATE, fileno(file), 0);
    if (!t->model) {
        goto read_failure;
    }

    // Success: Return control flow
    fclose(file);
    return true;

    // Failure: Break control flow
read_failure:
    fclose(file);
open_failure:
    return false;
}

/**
 * @section Model Params
 * @{
 */

/// @todo Consolidate repeated calculations into params.
/// @note These should only be done once during initialization.
///       Otherwise the repeated computations just compound one another over time.
bool model_read_params(Transformer* t, int override_seq_len) {
    if (!t) {
        return false;
    }

    memcpy(&t->params, t->model, sizeof(Params));
    if (QWEN_MAGIC != t->params.magic || QWEN_VERSION != t->params.version) {
        return false;
    }

    if (override_seq_len && override_seq_len <= t->params.seq_len) {
        t->params.seq_len = override_seq_len;
    }

    GS = t->params.group_size;
    return true;
}

/**
 * @section Model Weights
 */

/**
 * @brief Initialize and allocate quantized and fp32 weight tensors from memory-mapped stream.
 *
 * This function assumes `stream` points to a contiguous memory-mapped model checkpoint.
 * All fp32 weights (e.g. RMSNorm parameters) are read first, then the quantized tensors
 * are constructed using `q8_tensor_map`, which allocates memory internally and adjusts the stream
 * pointer.
 *
 * @param t        Pointer to Transformer model.
 * @return true on success, or false on error.
 */
bool model_read_weights(Transformer* t) {
    if (!t || !t->model || 0 == t->size) {
        return false;
    }

    /**
     * FP32 Weights (RMSNorms + LayerNorms)
     * These are read directly from the stream without allocation.
     * Layout order must match export script.
     */
    Params* p = &t->params;
    Weights* w = &t->weights;
    float* weights = (float*) t->model;

    w->att_rms_norm = weights;
    weights += p->n_layers * p->dim;

    w->ffn_rms_norm = weights;
    weights += p->n_layers * p->dim;

    w->out_rms_norm = weights;
    weights += p->dim;

    w->q_rms_norm = weights;
    weights += p->n_layers * p->head_dim;

    w->k_rms_norm = weights;
    weights += p->n_layers * p->head_dim;

    /**
     * Advance stream to beginning of quantized weights.
     * q8_tensor_map allocates memory for Q8Tensor and updates the stream pointer.
     */
    t->model = (void*) weights;

    // Token embeddings (quantized + dequantized)
    w->qe = q8_tensor_map(&t->model, 1, p->vocab_size * p->dim); // allocates internally
    w->fe = calloc(p->vocab_size * p->dim, sizeof(float)); // explicit malloc (must be freed)
    if (!w->fe) {
        return false;
    }

    q8_dequantize(w->qe, w->fe, p->vocab_size * p->dim);

    /**
     * Attention weights
     * All tensors are shaped [n_layers, dim * out_features] for consistent layout.
     * Matmul kernels must handle reshaping internally.
     */
    const int proj_dim = p->n_heads * p->head_dim;
    const int kv_dim = p->n_kv_heads * p->head_dim;

    w->wq = q8_tensor_map(&t->model, p->n_layers, p->dim * proj_dim);
    w->wk = q8_tensor_map(&t->model, p->n_layers, p->dim * kv_dim);
    w->wv = q8_tensor_map(&t->model, p->n_layers, p->dim * kv_dim);
    w->wo = q8_tensor_map(&t->model, p->n_layers, proj_dim * p->dim); // [proj, dim] format

    /**
     * Feed-forward weights
     * All three MLP branches use [hidden_dim × dim] layout in export
     */
    const int hidden_dim = p->hidden_dim;

    w->w1 = q8_tensor_map(&t->model, p->n_layers, p->dim * hidden_dim); // w1(x)
    w->w2 = q8_tensor_map(&t->model, p->n_layers, hidden_dim * p->dim); // w2(silu ⊙ w3(x))
    w->w3 = q8_tensor_map(&t->model, p->n_layers, p->dim * hidden_dim); // w3(x)

    /**
     * Output classifier
     * If shared_classifier is true, reuse token embedding matrix
     * (tied weights). Otherwise, allocate separate output proj_dim.
     */
    w->cls = p->shared_classifier ? w->qe : q8_tensor_map(&t->model, 1, p->dim * p->vocab_size);

    return true;
}

void model_free_weights(Transformer* t) {
    if (!t) {
        return;
    }

    Params* p = &t->params;
    Weights* w = &t->weights;
    if (!p || !w) {
        return;
    }

    // Free Q8Tensor arrays (do NOT free .q or .s — those are memory-mapped!)
    free(w->qe);
    // Dequantized token embeddings were malloc'd
    free(w->fe);

    // Attention weights
    free(w->wq);
    free(w->wk);
    free(w->wv);
    free(w->wo);

    // Feed-forward weights
    free(w->w1);
    free(w->w2);
    free(w->w3);

    // Classifier weights: only free if not shared
    if (!p->shared_classifier) {
        free(w->cls);
    }
}

/**
 * @section Model Forward State
 */

bool model_create_state(Transformer* t) {
    if (!t) {
        return false;
    }

    Params* p = &t->params;
    ForwardState* s = &t->state;
    if (!p || !s) {
        return false;
    }

    const int hidden_dim = p->hidden_dim;
    const int proj_dim = p->n_heads * p->head_dim; // IO Features
    const int kv_dim = p->n_kv_heads * p->head_dim;
    const uint64_t cache_len = (uint64_t) p->n_layers * p->seq_len * kv_dim;

    assert(0 == proj_dim % GS && "proj_dim must be divisible by GS");
    assert(0 == hidden_dim % GS && "hidden_dim must be divisible by GS");
    assert(0 != cache_len && "cache_len must be greater than 0");

    // Residual stream and attention output
    s->x = calloc(p->dim, sizeof(float)); // persistent
    s->x_rms_norm = calloc(proj_dim, sizeof(float)); // scratch for norm/project

    // Attention workspace
    s->q = calloc(proj_dim, sizeof(float));
    s->k = NULL; // s->k and s->v are aliases into slices of k_cache and v_cache
    s->v = NULL; // They point to the current time step within layer 'l'
    s->scores = calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));

    // Key/value memory (shared memory with KV)
    s->k_cache = calloc(cache_len, sizeof(float));
    s->v_cache = calloc(cache_len, sizeof(float));

    // MLP
    s->mlp_in = calloc(hidden_dim, sizeof(float));
    s->mlp_gate = calloc(hidden_dim, sizeof(float));

    // qx.q stores int8_t quantized values of x_rms_norm (proj_dim dim)
    s->qx.q = calloc(proj_dim, sizeof(int8_t));
    // qx.s stores per-group scale factors (proj_dim / GS)
    s->qx.s = calloc(proj_dim / GS, sizeof(float));

    s->qh.q = calloc(hidden_dim, sizeof(int8_t));
    s->qh.s = calloc(hidden_dim / GS, sizeof(float));

    // Check for allocation failures
    if (!s->x || !s->x_rms_norm || !s->q || !s->scores || !s->logits || !s->k_cache
        || !s->v_cache || !s->mlp_in || !s->mlp_gate || !s->qx.q || !s->qx.s || !s->qh.q
        || !s->qh.s) {
        fprintf(stderr, "state_create: allocation failed!\n");
        return false;
    }

    size_t total_bytes = p->dim * 3 * sizeof(float) + // x, x_rms_norm
                         proj_dim * (2 * sizeof(float) + sizeof(int8_t)) + // q, x_rms_norm, qx.q
                         (proj_dim / GS) * sizeof(float) + // qx.s
                         hidden_dim * (2 * sizeof(float) + sizeof(int8_t)) + // mlp, mlp_gate, qh.q
                         (hidden_dim / GS) * sizeof(float) + // qh.s
                         p->n_heads * p->seq_len * sizeof(float) + // scores
                         p->vocab_size * sizeof(float) + // logits
                         2 * cache_len * sizeof(float); // kv_cache
    fprintf(stderr, "state_create: allocated %.2f MB\n", total_bytes / (1024.0 * 1024.0));

    return true;
}

void model_free_state(Transformer* t) {
    if (!t) {
        return;
    }

    ForwardState* s = &t->state;
    if (!s) {
        return;
    }

    // Residual stream and attention output
    free(s->x);
    free(s->x_rms_norm);

    // Attention workspace
    free(s->q);
    free(s->scores);
    free(s->logits);

    // Key/value memory (shared memory with KV)
    free(s->k_cache);
    free(s->v_cache);

    // MLP
    free(s->mlp_in);
    free(s->mlp_gate);
    free(s->qx.q);
    free(s->qx.s);
    free(s->qh.q);
    free(s->qh.s);
}

/** @} */

/**
 * @section Transformer Model
 */

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
 * @param override_seq_len  Optional context length override (0 to use checkpoint default).
 * @return Pointer to a fully constructed Transformer, or NULL on failure.
 */
Transformer* transformer_create(const char* path, int override_seq_len) {
    if (!path) {
        return NULL;
    }

    Transformer* t = calloc(1, sizeof(Transformer));
    if (!t) {
        goto malloc_failure;
    }

    if (!model_read_checkpoint(t, path)) {
        goto read_failure;
    }

    void* model_base = t->model; // save the pointer
    if (!model_read_params(t, override_seq_len)) {
        goto read_failure;
    }

    if (!model_read_weights(t)) {
        goto read_failure;
    }

    if (!model_create_state(t)) {
        goto state_failure;
    }

    // Success: Return control flow
    t->model = model_base; // rewind to start
    return t;

    // Failure: Break control flow
state_failure:
    model_free_weights(t);
read_failure:
    free(t);
malloc_failure:
    return NULL;
}

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
 * @param t Pointer to a Transformer created by `transformer_create`.
 */
void transformer_free(Transformer* t) {
    if (!t) {
        return;
    }

    model_free_state(t);
    model_free_weights(t);
    munmap(t->model, t->size);
    free(t);
}

/** @} */

/**
 * @section Forward-pass
 * @{
 */

void rmsnorm(float* out, float* x, float* w, int size) {
    // calculate sum of squares
    float sos = 0.0f;

#pragma omp parallel for reduction(+ : sos)
    for (int i = 0; i < size; i++) {
        sos += x[i] * x[i];
    }

    sos = 1.0f / sqrtf((sos / size) + 1e-6f);

// normalize and scale
#pragma omp parallel for
    for (int i = 0; i < size; i++) {
        out[i] = w[i] * (sos * x[i]);
    }
}

/// @todo Cache max_val if last token's softmax is reused (e.g. in KV cache).
/// @todo Replace expf() with fast_approx_exp() if precision tradeoff is acceptable.
/// @todo Consider log-sum-exp for log-domain ops.
void softmax(float* x, int size) {
    // find max value (for numerical stability)
    float max_val = x[0]; // use first element as initial guess
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }

    // exponentiate and sum
    float sum = 0.0f;
#pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

// normalize
#pragma omp parallel for
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

void matmul(float* out, Q8Tensor* x, Q8Tensor* w, int n, int d) {
// W (d,n) @ x (n,) -> y (d,)
// For each output dimension i ∈ [0, d)
#pragma omp parallel for
    for (int i = 0; i < d; i++) {
        float val = 0;
        int off = i * n; // row start offset in flat W.q[]

        // Inner dot product in groups of GS
        for (int j = 0; j <= n - GS; j += GS) {
            int32_t dot = 0;
            for (int k = 0; k < GS; k++) {
                dot += x->q[j + k] * w->q[off + j + k];
            }

            val += ((float) dot) // Dequantize this group
                   * w->s[(off + j) / GS] // weight group scale
                   * x->s[j / GS]; // input group scale
        }

        out[i] = val;
    }
}

/// @todo Add precomputed rotary cache.
void rotary(float* x, int head_dim, int pos) {
    int half_dim = head_dim / 2;

#pragma omp parallel for
    for (int i = 0; i < half_dim; i++) {
        float angle = pos * powf(1e6f, -(float) i / half_dim);
        float cos_a = cosf(angle), sin_a = sinf(angle);

        float real = x[i];
        float imag = x[i + half_dim];

        x[i] = real * cos_a - imag * sin_a;
        x[i + half_dim] = real * sin_a + imag * cos_a;
    }
}

/// @brief σ(x) = 1 / 1 + exp(-x)
/// @note This can not be modified. The model will become incoherent.
float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

/// @brief x∗σ(x), where σ(x) is the logistic sigmoid.
float silu(float x) {
    return x * sigmoid(x);
}

/// @brief SwiGLU(x) = silu(W₁x) ⊙ W₃x
/// @note Modifying this math will corrupt model behavior.
///       Make sure SiLU and element-wise multiplication are preserved.
void swiglu(float* x1, float* x3, int size) {
#pragma omp parallel for
    for (int i = 0; i < size; i++) {
        x1[i] = silu(x1[i]) * x3[i];
    }
}

void attention(Transformer* t, int l, int pos) {
    Params* p = &t->params;
    ForwardState* s = &t->state;

    int head_dim = p->head_dim;
    int kv_mul = p->n_heads / p->n_kv_heads; // multi-query attention
    int kv_dim = p->n_kv_heads * head_dim;
    uint64_t loff = (uint64_t) l * p->seq_len * kv_dim;

    for (int h = 0; h < p->n_heads; h++) {
        float* q = s->q + h * head_dim;
        float* scores = s->scores + h * p->seq_len;
        float* head_out = s->x_rms_norm + h * head_dim;

// Compute attention scores
#pragma omp parallel for
        for (int i = 0; i <= pos; i++) {
            float* k = s->k_cache + loff + i * kv_dim + (h / kv_mul) * head_dim;
            float score = 0.0f;
            for (int j = 0; j < head_dim; j++) {
                score += q[j] * k[j];
            }

            scores[i] = score / sqrtf((float) head_dim);
        }

        // Normalize scores to get attention weights
        softmax(scores, pos + 1);

        // Initialize output vector for this head
        memset(head_out, 0, sizeof(float) * head_dim);

// Accumulate weighted values in thread-safe way
#pragma omp parallel
        {
            float tmp[head_dim];
            memset(tmp, 0, head_dim * sizeof(float));

#pragma omp for
            for (int i = 0; i <= pos; i++) {
                float* v = s->v_cache + loff + i * kv_dim + (h / kv_mul) * head_dim;
                for (int j = 0; j < head_dim; j++) {
                    tmp[j] += scores[i] * v[j];
                }
            }

            // Reduce thread-local buffer into shared output
            for (int i = 0; i < head_dim; i++) {
#pragma omp atomic
                head_out[i] += tmp[i];
            }
        }
    }
}

/**
 *             ┌────────────┐
 * token ────▶│ embeddings │
 *             └────────────┘
 *                   ▼
 *  ┌────────────────────────────────────────┐
 *  │          Per Layer (32x)               │
 *  │                                        │
 *  │ x ─ RMSNorm ─ QKV ─ Rotary ─ Attention │
 *  │                │               │       │
 *  │                ▼               ▼       │
 *  │           FFN w1/w3        Softmax · V │
 *  │                │               │       │
 *  │            SwiGLU <─── Scores ─┘       │
 *  │                ▼                       │
 *  │           FFN w2 → add to x            │
 *  └────────────────────────────────────────┘
 *                   ▼
 *            ┌───────────────┐
 *            | Final RMSNorm |
 *            └───────────────┘
 *                   ▼
 *             ┌────────────┐
 *             | Classifier |
 *             └────────────┘
 *                   ▼
 *                 logits
 */
float* forward(Transformer* t, int token, int pos) {
    Params* p = &t->params;
    Weights* w = &t->weights;
    ForwardState* s = &t->state;

    int kv_dim = p->n_kv_heads * p->head_dim;
    int hidden_dim = p->hidden_dim;
    int proj_dim = p->n_heads * p->head_dim;

    /**
     * Input Embedding
     * Load the embedding vector for the input token into x (residual stream)
     */
    memcpy(s->x, w->fe + token * p->dim, p->dim * sizeof(float));

    /**
     * Layer Loop
     */
    for (int l = 0; l < p->n_layers; l++) {
        // KV cache layer offset for current layer and position
        uint64_t loff = l * (uint64_t) p->seq_len * kv_dim;

        // Slice the cache for this time step
        s->k = s->k_cache + loff + pos * kv_dim;
        s->v = s->v_cache + loff + pos * kv_dim;

        /**
         * Attention RMSNorm
         * Normalize the input x before computing Q/K/V
         */
        rmsnorm(s->x_rms_norm, s->x, w->att_rms_norm + l * p->dim, p->dim);

        /**
         * Q/K/V Projection (Quantized Matmuls)
         */
        q8_quantize(&s->qx, s->x_rms_norm, p->dim);
        matmul(s->q, &s->qx, w->wq + l, p->dim, proj_dim); // Q
        matmul(s->k, &s->qx, w->wk + l, p->dim, kv_dim); // K
        matmul(s->v, &s->qx, w->wv + l, p->dim, kv_dim); // V

        /**
         * Q & K RMSNorm + Rotary Embedding
         */
        float* gq = w->q_rms_norm + l * p->head_dim;
        float* gk = w->k_rms_norm + l * p->head_dim;

        for (int h = 0; h < p->n_heads; h++) {
            float* q = s->q + h * p->head_dim;
            rmsnorm(q, q, gq, p->head_dim); // Normalize each query head
            rotary(q, p->head_dim, pos); // Apply positional rotation
        }

        for (int h = 0; h < p->n_kv_heads; h++) {
            float* k = s->k + h * p->head_dim;
            rmsnorm(k, k, gk, p->head_dim); // Normalize each key head
            rotary(k, p->head_dim, pos); // Apply positional rotation
        }

        /**
         * Multi-Head Attention Computation
         */
        attention(t, l, pos); // Uses Q/K/V to compute context → written to x_rms_norm

        /**
         * Output Projection + Residual Add
         */
        q8_quantize(&s->qx, s->x_rms_norm, proj_dim);
        matmul(s->x_rms_norm, &s->qx, w->wo + l, proj_dim, p->dim);
        for (int i = 0; i < p->dim; i++) {
            s->x[i] += s->x_rms_norm[i]; // Add attention output back into residual stream
        }

        /**
         * FFN RMSNorm
         */
        rmsnorm(s->x_rms_norm, s->x, w->ffn_rms_norm + l * p->dim, p->dim);

        /**
         * FFN Input Projections (w1 and w3)
         */
        q8_quantize(&s->qx, s->x_rms_norm, p->dim);
        matmul(s->mlp_in, &s->qx, w->w1 + l, p->dim, hidden_dim); // w1(x)
        matmul(s->mlp_gate, &s->qx, w->w3 + l, p->dim, hidden_dim); // w3(x)

        /**
         * SwiGLU Activation
         */
        swiglu(s->mlp_in, s->mlp_gate, hidden_dim); // mlp_in = silu(w1) * w3

        /**
         * FFN Output Projection + Residual Add
         */
        q8_quantize(&s->qh, s->mlp_in, hidden_dim);
        matmul(s->x_rms_norm, &s->qh, w->w2 + l, hidden_dim, p->dim);
        for (int i = 0; i < p->dim; i++) {
            s->x[i] += s->x_rms_norm[i]; // Add FFN output into residual stream
        }
    }

    /**
     * Final LayerNorm + Output Projection
     */
    rmsnorm(s->x, s->x, w->out_rms_norm, p->dim); // Final norm before logits

    q8_quantize(&s->qx, s->x, p->dim);
    matmul(s->logits, &s->qx, w->cls, p->dim, p->vocab_size); // Final linear classifier
    return s->logits;
}

/** @} */

/**
 * @section Tokenizer Model
 * @brief Byte Pair Encoding (BPE) Tokenizer to map strings and tokens.
 */

typedef struct Template {
    char* data;     // dynamically allocated UTF-8 string
    ssize_t size;   // number of bytes read (excluding null terminator)
} Template;

typedef struct Token {
    char* entry;     // null-terminated UTF-8 token
    float score;    // merge rank or base token marker
} Token;

typedef struct Tokenizer {
    Token* tokens; // token → string + score
    Template* prompt;    // user-only prompt
    Template* system;    // system + user prompt
    int bos_id;
    int eos_id;
    int vocab_size;
    int max_token_length;
} Tokenizer;

Template* template_create(char* in_file, int enable_system, int enable_thinking) {
    const char* suffix = NULL;
    if (enable_system) {
        suffix = enable_thinking ? ".template.with-system-and-thinking" : ".template.with-system";
    } else {
        suffix = enable_thinking ? ".template.with-thinking" : ".template";
    }

    // Construct full file path
    size_t len = strlen(in_file) + strlen(suffix);
    char* file_path = calloc(len + 1, 1);
    if (!file_path) return NULL;

    memcpy(file_path, in_file, strlen(in_file));
    memcpy(file_path + strlen(in_file), suffix, strlen(suffix));
    file_path[len] = '\0';

    FILE* file = fopen(file_path, "rb");
    free(file_path); // cleanup here is safe
    if (!file) return NULL;

    fseek(file, 0, SEEK_END);
    ssize_t size = ftell(file);
    rewind(file);

    Template* template = calloc(1, sizeof(Template));
    if (!template) { fclose(file); return NULL; }

    template->size = size;
    template->data = calloc(size + 1, 1); // null-terminate for convenience
    if (!template->data) {
        fclose(file);
        free(template);
        return NULL;
    }

    fread(template->data, 1, size, file);
    fclose(file);
    return template;
}

void template_free(Template* t) {
    if (!t) return;
    free(t->data);
    free(t);
}

void build_tokenizer(Tokenizer *t, char *checkpoint_path, int vocab_size, int enable_thinking) {
    char tokenizer_path[1024];

    strcpy(tokenizer_path, checkpoint_path);
    strcat(tokenizer_path, ".tokenizer");

    t->vocab_size = vocab_size;
    // malloc space to hold the scores and the strings
    t->vocab = (char **)malloc(vocab_size * sizeof(char *));
    t->merge_scores = (float *)malloc(vocab_size * sizeof(float));

    // read in the file
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file) { fprintf(stderr, "Couldn't load tokenizer model %s\n", tokenizer_path); exit(EXIT_FAILURE); }
    fread(&t->max_token_length, sizeof(int), 1, file);
    fread(&t->bos_token_id, sizeof(int), 1, file);
    fread(&t->eos_token_id, sizeof(int), 1, file);

    int len;

    for (int i = 0; i < vocab_size; i++) {
        if (fread(t->merge_scores + i, sizeof(float), 1, file) != 1) {
            t->vocab[i] = (char *)malloc(1);
            t->vocab[i][0] = '\0'; // add the string terminating token
        } else {
            fread(&len, sizeof(int), 1, file);
            t->vocab[i] = (char *)malloc(len + 1);
            fread(t->vocab[i], 1, len, file);
            t->vocab[i][len] = '\0'; // add the string terminating token
        }
    }
    fclose(file);

    load_prompt_template(checkpoint_path, t->prompt_template, 0, enable_thinking);
    load_prompt_template(checkpoint_path, t->system_prompt_template, 1, enable_thinking);
}

void free_tokenizer(Tokenizer *t) {
    for (int i = 0; i < t->vocab_size; i++) { free(t->vocab[i]); }
    free(t->vocab);
    free(t->merge_scores);
}

char *decode(Tokenizer *t, int token) {
    return t->vocab[token];
}

int str_lookup(char *str, char **vocab, int vocab_size) {
    // find a match for str in vocab, return its index or -1 if not found
    for (int i = 0; i < vocab_size; i++)
        if (!strcmp(str, vocab[i]))
            return i;

    return -1;
}

void encode(Tokenizer *t, char *text, int *tokens, int *n_tokens) {
    // encode the string text (input) into an upper-bound preallocated tokens[] array

    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    char *str_buffer = malloc((t->max_token_length*2 +1 +2) * sizeof(char));
    char special_token[64 + 1];

    // start at 0 tokens
    *n_tokens = 0;

    // process the raw (UTF-8) byte sequence of the input string
    for (char *c = text; *c != '\0'; c++) {
        int id, found_special_token = 0;

        // set the buffer to the current byte
        str_buffer[0] = *c;
        str_buffer[1] = '\0';

        // special tokens begin with < and end with >. If we find a substring beginning with <
        // and ending with > and there's a token in the vocab for it, use that instead of parsing into
        // shorter tokens
        if (*c == '<') {
          int end_of_token_pos = -1;
          found_special_token = 0;
          for (int k = 0; *c != '\0' && k < 64; k++) {
              if (c[k] == '>') {
                  end_of_token_pos = k;
                  break;
              }
          }

          if (end_of_token_pos != -1) {
              strncpy(special_token, c, end_of_token_pos + 1);
              special_token[end_of_token_pos + 1] = 0;

              id = str_lookup(special_token, t->vocab, t->vocab_size);
              if (id != -1) {
                  c += end_of_token_pos;
                  found_special_token = 1;
              }
          }
        }

        // not a special token, just look up the single character
        if (!found_special_token)
            id = str_lookup(str_buffer, t->vocab, t->vocab_size);

        if (id != -1) {
            // we found this codepoint in vocab, add it as a token
            tokens[(*n_tokens)++] = id;
        } else {
            printf("Warning: unknown character code point %d in input, skipping.\n", *str_buffer);
            (*n_tokens)++;
        }
    }

    // merge the best consecutive pair each iteration
    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i = 0; i < (*n_tokens - 1); i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i + 1]]);
            int id = str_lookup(str_buffer, t->vocab, t->vocab_size);

            if (id != -1 && t->merge_scores[id] > best_score) {
                // this merge pair exists in vocab! record its score and position
                best_score = t->merge_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1)
            break; // we couldn't find any more pairs to merge, so we're done

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for (int i = best_idx + 1; i < (*n_tokens - 1); i++)
            tokens[i] = tokens[i + 1];

        (*n_tokens)--; // token length decreased
    }

    free(str_buffer);
}

/** @} */

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

typedef struct {
    float prob;
    int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

typedef struct {
    int vocab_size;
    ProbIndex *probindex; // buffer used in top-p sampling
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

int sample_argmax(float *probabilities, int n) {
    // return the index that has the highest probability
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

int sample_mult(float *probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf)
            return i;
    }
    return n - 1; // in case of rounding errors
}

int compare(const void *a, const void *b) {
    ProbIndex *a_ = (ProbIndex *) a;
    ProbIndex *b_ = (ProbIndex *) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

int sample_topp(float *probabilities, int n, float topp, ProbIndex *probindex, float coin) {
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".
    // coin is a random number in [0, 1), usually from random_f32()

    int n0 = 0;
    // quicksort indices in descending order of probabilities
    // values smaller than (1 - topp) / (n - 1) cannot be part of the result
    // so for efficiency we crop these out as candidates before sorting
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare);

    // truncate the list where cumulative probability exceeds topp
    float cumulative_prob = 0;
    int last_idx = n0 - 1; // in case of rounding errors consider all elements
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break; // we've exceeded topp by including last_idx
        }
    }

    // sample from the truncated list
    float r = coin * cumulative_prob;
    float cdf = 0;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf)
            return probindex[i].index;
    }
    return probindex[last_idx].index; // in case of rounding errors
}

void build_sampler(Sampler *sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    // buffer only used with nucleus sampling; may not need but it's ~small
    sampler->probindex = malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler *sampler) {
    free(sampler->probindex);
}

unsigned int random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler *sampler, float *logits) {
    // sample the token given the logits and some hyperparameters
    int next;
    if (sampler->temperature == 0) {
        // greedy argmax sampling: take the token with the highest probability
        next = sample_argmax(logits, sampler->vocab_size);
    } else {
        // apply the temperature to the logits
        for (int q=0; q<sampler->vocab_size; q++) { logits[q] /= sampler->temperature; }
        // apply softmax to the logits to get the probabilities for next token
        softmax(logits, sampler->vocab_size);
        // flip a (float) coin (this is our source of entropy for sampling)
        float coin = random_f32(&sampler->rng_state);
        // we sample from this distribution to get the next token
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            // simply sample from the predicted probability distribution
            next = sample_mult(logits, sampler->vocab_size, coin);
        } else {
            // top-p (nucleus) sampling, clamping the least likely tokens to zero
            next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
    return next;
}

// ----------------------------------------------------------------------------
// generation loop

void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt) {
    char *empty_prompt = "";
    if (prompt == NULL) { prompt = empty_prompt; }

    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    int *prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
    encode(tokenizer, prompt, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "Please provide a prompt using -i <string> on the command line.\n");
        exit(EXIT_FAILURE);
    }

    // start the main loop
    int next;        // will store the next token in the sequence
    int token = prompt_tokens[0]; // kick off with the first token in the prompt
    int pos = 0;     // position in the sequence

    while (pos < transformer->params.seq_len) {
        // forward the transformer to get logits for the next token
        float *logits = forward(transformer, token, pos);

        // advance the state state machine
        if (pos < num_prompt_tokens - 1) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos + 1];
        } else {
            // otherwise sample the next token from the logits
            next = sample(sampler, logits);
        }
        pos++;

        // print the token as string, decode it with the Tokenizer object
        printf("%s", decode(tokenizer, token));
        fflush(stdout);
        token = next;

        // data-dependent terminating condition: the BOS token delimits sequences
        if (pos >= num_prompt_tokens && (next == tokenizer->bos_token_id || next == tokenizer->eos_token_id))
            break;
    }
    printf("\n");
    free(prompt_tokens);
}

void read_stdin(const char *guide, char *buffer, size_t bufsize) {
    // read a line from stdin, up to but not including \n
    printf("%s", guide);
    if (fgets(buffer, bufsize, stdin) != NULL) {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n')
            buffer[len - 1] = '\0'; // strip newline
    }
}

// ----------------------------------------------------------------------------
// chat loop

void chat(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *cli_user_prompt, char *system_prompt) {
    // buffers for reading the system prompt and user prompt from stdin
    char user_prompt[32768];
    char rendered_prompt[32768];
    int num_prompt_tokens = 0;
    int *prompt_tokens = (int *)malloc(32768 * sizeof(int));
    int user_idx;

    // start the main loop
    int8_t user_turn = 1; // user starts
    int next;        // will store the next token in the sequence
    int token;       // stores the current token to feed into the transformer
    // int prev_token;
    int pos = 0;     // position in the sequence

    while (1) {
        // if context window is exceeded, clear it
        if (pos >= transformer->params.seq_len) {
            printf("\n(context window full, clearing)\n");
            user_turn = 1;
            pos = 0;
        }

        // when it is the user's turn to contribute tokens to the dialog...
        if (user_turn) {
            // get the user prompt
            if (cli_user_prompt != NULL) {
                // user prompt for position 0 was passed in, use it
                if (pos > 0)
                    break;
                strcpy(user_prompt, cli_user_prompt);
            } else {
                // otherwise get user prompt from stdin
                read_stdin("\n> ", user_prompt, sizeof(user_prompt));
                // terminate if user enters a blank prompt
                if (!user_prompt[0])
                    break;
            }

            // render user/system prompts into the Qwen3 prompt template schema
            if (pos == 0 && system_prompt)
                sprintf(rendered_prompt, tokenizer->system_prompt_template, system_prompt, user_prompt);
            else
                sprintf(rendered_prompt, tokenizer->prompt_template, user_prompt);

            // encode the rendered prompt into tokens
            encode(tokenizer, rendered_prompt, prompt_tokens, &num_prompt_tokens);
            user_idx = 0; // reset the user index
            user_turn = 0;
        }

        // determine the token to pass into the transformer next
        if (user_idx < num_prompt_tokens) {
            // if we are still processing the input prompt, force the next prompt token
            token = prompt_tokens[user_idx++];
        } else {
            // otherwise use the next token sampled from previous turn
            token = next;
        }

        // printf("|pos=%d token=%d '%s'|\n",pos,token,tokenizer->vocab[token]);

        // forward the transformer to get logits for the next token
        float *logits = forward(transformer, token, pos);
        next = sample(sampler, logits);
        pos++;

        // assistant is responding
        if (user_idx >= num_prompt_tokens) {
            if (token == tokenizer->bos_token_id || token == tokenizer->eos_token_id) {
                // EOS token ends the assistant turn
                printf("\n");
                user_turn = 1;
            } else if (next != tokenizer->bos_token_id && next != tokenizer->eos_token_id) {
                printf("%s", decode(tokenizer, next));
                fflush(stdout);
            }
        }
    }
    free(prompt_tokens);
}

// ----------------------------------------------------------------------------
// CLI

void error_usage() {
    fprintf(stderr, "Usage:   runq <checkpoint> [options]\n");
    fprintf(stderr, "Example: runq Qwen3-4B.bin -r 1\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1], default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -c <int>    context window size, 0 (default) = max_seq_len\n");
    fprintf(stderr, "  -m <string> mode: generate|chat, default: chat\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -y <string> system prompt in chat mode, default is none\n");
    fprintf(stderr, "  -r <int>    reasoning mode, 0 (default) = no thinking, 1 = thinking\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {
    // default parameters
    char *checkpoint_path = NULL;  // e.g. out/model.bin
    float temperature = 1.0f;   // 0 = greedy deterministic. 1.0 = original. don't set higher
    float topp = 0.9f;          // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    char *prompt = NULL;        // prompt string
    unsigned long long rng_seed = 0; // seed rng with time by default
    char *mode = "chat";        // generate|chat
    char *system_prompt = NULL; // the (optional) system prompt to use in chat mode
    int enable_thinking = 0;    // 1 enables thinking
    int ctx_length = 0;         // context length

    // poor man's C argparse so we can override the defaults above from the command line
    if (argc >= 2) { checkpoint_path = argv[1]; } else { error_usage(); }
    for (int i = 2; i < argc; i+=2) {
        // do some basic validation
        if (i + 1 >= argc) { error_usage(); } // must have arg after flag
        if (argv[i][0] != '-') { error_usage(); } // must start with dash
        if (strlen(argv[i]) != 2) { error_usage(); } // must be -x (one dash, one letter)
        // read in the args
        if (argv[i][1] == 't') { temperature = atof(argv[i + 1]); }
        else if (argv[i][1] == 'p') { topp = atof(argv[i + 1]); }
        else if (argv[i][1] == 's') { rng_seed = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'c') { ctx_length = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'i') { prompt = argv[i + 1]; }
        else if (argv[i][1] == 'm') { mode = argv[i + 1]; }
        else if (argv[i][1] == 'y') { system_prompt = argv[i + 1]; }
        else if (argv[i][1] == 'r') { enable_thinking = atoi(argv[i + 1]); }
        else { error_usage(); }
    }

    // parameter validation/overrides
    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0) temperature = 0;
    if (topp < 0.0f || 1.0f < topp) topp = 0.9f;

    // build the Transformer via the model .bin file
    Transformer* transformer = transformer_create(checkpoint_path, ctx_length);

    // build the Tokenizer via the tokenizer .bin file
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, checkpoint_path, transformer->params.vocab_size, enable_thinking);

    // build the Sampler
    Sampler sampler;
    build_sampler(&sampler, transformer->params.vocab_size, temperature, topp, rng_seed);

    if (!prompt) {
        printf(
            "hidden_size=%d, "
            "intermediate_size=%d, "
            "num_hidden_layers=%d, "
            "num_attention_heads=%d, "
            "num_kv_heads=%d, "
            "head_dim=%d, "
            "ctx_length=%d, "
            "vocab_size=%d, "
            "shared_classifier=%d, "
            "quantization_block_size=%d\n",
            transformer->params.dim,
            transformer->params.hidden_dim,
            transformer->params.n_layers,
            transformer->params.n_heads,
            transformer->params.n_kv_heads,
            transformer->params.head_dim,
            transformer->params.seq_len,
            transformer->params.vocab_size,
            transformer->params.shared_classifier,
            transformer->params.group_size);
    }

    // run!
    if (strcmp(mode, "generate") == 0) {
        generate(transformer, &tokenizer, &sampler, prompt);
    } else if (strcmp(mode, "chat") == 0) {
        chat(transformer, &tokenizer, &sampler, prompt, system_prompt);
    } else {
        fprintf(stderr, "Unknown mode: %s\n", mode);
        error_usage();
    }

    // memory and file handles cleanup
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    transformer_free(transformer);
    return 0;
}
