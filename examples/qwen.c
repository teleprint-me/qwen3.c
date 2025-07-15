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
 * @ref https://en.wikipedia.org/wiki/Ship_of_Theseus
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
/// Requires ASAN_OPTIONS=detect_odr_violation=0 to disable ODR complaints.
/// This happens because the library is linked to this binary when it's built
/// and it's defined in the Q8 header.
int GS = 2; // global quantization group size

/**
 * @struct Q8Tensor
 * @brief A quantized tensor with int8 values and per-group scale factors.
 */
typedef struct Q8Tensor {
    float* s; ///< scaling factors per group
    int8_t* q; ///< quantized values
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
        float* xg = x + group * GS;
        int8_t* qg = qt->q + group * GS;

        // Find max absolute value
        float wmax = fabsf(xg[0]);
#pragma omp simd reduction(max : wmax)
        for (int i = 1; i < GS; i++) {
            wmax = fmaxf(wmax, fabsf(xg[i]));
        }

        float scale = (wmax == 0.0f) ? 1e-6f : (wmax / Q_MAX); // avoid div by 0
        qt->s[group] = scale;

/// @note Clamp to [-127, 127] to avoid int8 overflow on rare large values.
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
Q8Tensor* q8_tensor_map(void** buffer, int n, int size) {
    if (!buffer || n <= 0 || size <= 0) {
        return NULL;
    }

    void* cursor = *buffer; // current pos
    Q8Tensor* qt = calloc(n, sizeof(Q8Tensor));
    if (!qt) {
        return NULL;
    }

    // Buffer must be read linearly
    for (int i = 0; i < n; i++) {
        // map q8 values
        qt[i].q = (int8_t*) cursor;
        cursor = (int8_t*) cursor + size;

        // map scalars
        qt[i].s = (float*) cursor;
        cursor = (float*) cursor + size / GS;
    }

    *buffer = cursor; // advance buf
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

    const int padding = 256;
    t->model = ((char*) t->model) + padding; // add alignment to weights

    fprintf(stderr, "[Params] magic=%s\n", (char*) &t->params.magic);
    fprintf(stderr, "[Params] version=%d\n", t->params.version);
    fprintf(stderr, "[Params] hidden_size=%d\n", t->params.dim);
    fprintf(stderr, "[Params] intermediate_size=%d\n", t->params.hidden_dim);
    fprintf(stderr, "[Params] num_hidden_layers=%d\n", t->params.n_layers);
    fprintf(stderr, "[Params] num_attention_heads=%d\n", t->params.n_heads);
    fprintf(stderr, "[Params] num_kv_heads=%d\n", t->params.n_kv_heads);
    fprintf(stderr, "[Params] vocab_size=%d\n", t->params.vocab_size);
    fprintf(stderr, "[Params] seq_len=%d\n", t->params.seq_len);
    fprintf(stderr, "[Params] head_dim=%d\n", t->params.head_dim);
    fprintf(stderr, "[Params] shared_classifier=%d\n", t->params.shared_classifier);
    fprintf(stderr, "[Params] group_size=%d\n", t->params.group_size); // block size

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

    size_t total_bytes =
        // FP32 weights
        p->n_layers * p->dim * 2 * sizeof(float) + // att + ffn
        p->dim * sizeof(float) + // out
        p->n_layers * p->head_dim * 2 * sizeof(float) + // q and k

        // Token Embeddings
        p->vocab_size * p->dim * sizeof(int8_t) + // qe.q
        (p->vocab_size * p->dim / GS) * sizeof(float) + // qe.s
        p->vocab_size * p->dim * sizeof(float) + // fe

        // Attention weights
        2 * p->n_layers * p->dim * proj_dim * sizeof(int8_t) + // wq, wo (q)
        2 * p->n_layers * (p->dim * proj_dim / GS) * sizeof(float) + // wq, wo (s)
        2 * p->n_layers * p->dim * kv_dim * sizeof(int8_t) + // wk, wv (q)
        2 * p->n_layers * (p->dim * kv_dim / GS) * sizeof(float) + // wk, wv (s)

        // Feedforward weights
        3 * p->n_layers * p->dim * hidden_dim * sizeof(int8_t) + // w1, w2, w3 (q)
        3 * p->n_layers * (p->dim * hidden_dim / GS) * sizeof(float); // w1, w2, w3 (s)

    if (!p->shared_classifier) {
        total_bytes += p->dim * p->vocab_size * sizeof(int8_t); // cls.q
        total_bytes += (p->dim * p->vocab_size / GS) * sizeof(float); // cls.s
    }

    fprintf(stderr, "[Weights] Allocated %.2f MB\n", total_bytes / (1024.0 * 1024.0));

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

    const int proj_dim = p->n_heads * p->head_dim; // IO Features
    const int kv_dim = p->n_kv_heads * p->head_dim;
    const uint64_t cache_len = (uint64_t) p->n_layers * p->seq_len * kv_dim;

    assert(0 == proj_dim % GS && "proj_dim must be divisible by GS");
    assert(0 == p->hidden_dim % GS && "hidden_dim must be divisible by GS");
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
    s->mlp_in = calloc(p->hidden_dim, sizeof(float));
    s->mlp_gate = calloc(p->hidden_dim, sizeof(float));

    s->qx = (Q8Tensor) {
        // int8_t quantized values of x_rms_norm (proj_dim)
        .q = calloc(proj_dim, sizeof(int8_t)),
        // per-group scale factors (proj_dim / GS)
        .s = calloc(proj_dim / GS, sizeof(float)),
    };

    s->qh = (Q8Tensor) {
        .q = calloc(p->hidden_dim, sizeof(int8_t)),
        .s = calloc(p->hidden_dim / GS, sizeof(float)),
    };

    // Check for allocation failures
    if (!s->x || !s->x_rms_norm || !s->q || !s->scores || !s->logits || !s->k_cache || !s->v_cache
        || !s->mlp_in || !s->mlp_gate || !s->qx.q || !s->qx.s || !s->qh.q || !s->qh.s) {
        fprintf(stderr, "[ForwardState] Allocation failed!\n");
        return false;
    }

    size_t total_bytes = p->dim * 3 * sizeof(float) + // x, x_rms_norm
                         proj_dim * (2 * sizeof(float) + sizeof(int8_t)) + // q, x_rms_norm, qx.q
                         (proj_dim / GS) * sizeof(float) + // qx.s
                         p->hidden_dim * (2 * sizeof(float) + sizeof(int8_t)) + // mlp, mlp_gate, qh.q
                         (p->hidden_dim / GS) * sizeof(float) + // qh.s
                         p->n_heads * p->seq_len * sizeof(float) + // scores
                         p->vocab_size * sizeof(float) + // logits
                         2 * cache_len * sizeof(float); // kv_cache

    fprintf(stderr, "[ForwardState] Allocated %.2f MB\n", total_bytes / (1024.0 * 1024.0));

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
        // debug
        if (isnan(x[i]) || isinf(x[i])) {
            fprintf(stderr, "[Softmax] ⚠️  Invalid input: x[%d] = %f\n", i, x[i]);
        }

        if (x[i] > max_val) {
            max_val = x[i];
        }
    }

    // exponentiate and sum
    float sum = 0.0f;
#pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i < size; i++) {
        // debug
        float exp_val = expf(x[i] - max_val);
        if (isnan(exp_val) || isinf(exp_val)) {
            fprintf(stderr, "[Softmax] NaN/Inf at i=%d: x=%f max_val=%f\n", i, x[i], max_val);
        }
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
        matmul(s->mlp_in, &s->qx, w->w1 + l, p->dim, p->hidden_dim); // w1(x)
        matmul(s->mlp_gate, &s->qx, w->w3 + l, p->dim, p->hidden_dim); // w3(x)

        /**
         * SwiGLU Activation
         */
        swiglu(s->mlp_in, s->mlp_gate, p->hidden_dim); // mlp_in = silu(w1) * w3

        /**
         * FFN Output Projection + Residual Add
         */
        q8_quantize(&s->qh, s->mlp_in, p->hidden_dim);
        matmul(s->x_rms_norm, &s->qh, w->w2 + l, p->hidden_dim, p->dim);
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
 * @section Tokenizer: Chat Template
 * @{
 */

typedef struct Template {
    char* data; ///< UTF-8 prompt string (dynamically allocated)
    ssize_t size; ///< Size in bytes (excluding null terminator)
} Template;

Template* template_create(const char* in_file, int enable_system, int enable_thinking) {
    const char* suffix = NULL;
    if (enable_system) {
        suffix = enable_thinking ? ".template.with-system-and-thinking" : ".template.with-system";
    } else {
        suffix = enable_thinking ? ".template.with-thinking" : ".template";
    }

    // Construct full file path
    size_t len = strlen(in_file) + strlen(suffix);
    char* file_path = calloc(len + 1, 1);
    if (!file_path) {
        return NULL;
    }

    memcpy(file_path, in_file, strlen(in_file));
    memcpy(file_path + strlen(in_file), suffix, strlen(suffix));
    file_path[len] = '\0';

    FILE* file = fopen(file_path, "rb");
    free(file_path); // cleanup here is safe
    if (!file) {
        return NULL;
    }

    fseek(file, 0, SEEK_END);
    ssize_t size = ftell(file);
    rewind(file);

    Template* template = calloc(1, sizeof(Template));
    if (!template) {
        fclose(file);
        return NULL;
    }

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
    if (!t) {
        return;
    }
    free(t->data);
    free(t);
}

/** @} */

/**
 * @section Tokenizer: BPE (Byte-pair Encoding) NFC (Normalization Form Canonical Composition)
 * @{
 */

typedef struct Token {
    char* entry; ///< Null-terminated UTF-8 token
    float score; ///< Merge rank score (higher is better)
} Token;

typedef struct Tokenizer {
    Token* tokens; ///< Vocabulary table (id → token)
    Template* prompt; ///< User prompt template
    Template* system; ///< System + user prompt template
    int bos_id; ///< Beginning-of-sequence token id
    int eos_id; ///< End-of-sequence token id
    int vocab_size; ///< Total number of tokens
    int max_token_length; ///< Maximum UTF-8 length of any token
} Tokenizer;

Tokenizer* tokenizer_create(const char* in_file, int enable_thinking) {
    // Build file path for .tokenizer
    const char* suffix = ".tokenizer";
    size_t len = strlen(in_file) + strlen(suffix);
    char* file_path = calloc(len + 1, 1);
    if (!file_path) {
        return NULL;
    }

    memcpy(file_path, in_file, strlen(in_file));
    memcpy(file_path + strlen(in_file), suffix, strlen(suffix));

    FILE* file = fopen(file_path, "rb");
    if (!file) {
        fprintf(stderr, "[Tokenizer] Failed to open %s\n", file_path);
        free(file_path);
        return NULL;
    }
    free(file_path);

    // Read header
    uint32_t magic = 0;
    int32_t version = 0;
    fread(&magic, sizeof(uint32_t), 1, file);
    fread(&version, sizeof(int32_t), 1, file);

    if (QTKN_MAGIC != magic || QTKN_VERSION != version) {
        fprintf(stderr, "[Tokenizer] Invalid tokenizer format.\n");
        fclose(file);
        return NULL;
    }

    Tokenizer* t = calloc(1, sizeof(Tokenizer));
    if (!t) {
        fclose(file);
        return NULL;
    }

    fread(&t->vocab_size, sizeof(int32_t), 1, file);
    fread(&t->max_token_length, sizeof(int32_t), 1, file);
    fread(&t->bos_id, sizeof(int32_t), 1, file);
    fread(&t->eos_id, sizeof(int32_t), 1, file);

    t->tokens = calloc(t->vocab_size, sizeof(Token));
    if (!t->tokens) {
        fclose(file);
        free(t);
        return NULL;
    }

    // Read each token entry
    for (int i = 0; i < t->vocab_size; i++) {
        float score;
        int length;

        if (fread(&score, sizeof(float), 1, file) != 1) {
            fprintf(stderr, "[Tokenizer] Score read error at index %d\n", i);
            fclose(file);
            for (int k = 0; k < i; k++) {
                free(t->tokens[k].entry);
            }
            free(t->tokens);
            free(t);
            return NULL;
        }

        if (fread(&length, sizeof(int), 1, file) != 1) {
            fprintf(stderr, "[Tokenizer] Length read error at index %d\n", i);
            fclose(file);
            for (int k = 0; k < i; k++) {
                free(t->tokens[k].entry);
            }
            free(t->tokens);
            free(t);
            return NULL;
        }

        char* buffer = calloc(length + 1, 1);
        if (!buffer || fread(buffer, 1, length, file) != (size_t) length) {
            fprintf(stderr, "[Tokenizer] Token read error at index %d\n", i);
            if (buffer) {
                free(buffer);
            }
            fclose(file);
            for (int k = 0; k < i; k++) {
                free(t->tokens[k].entry);
            }
            free(t->tokens);
            free(t);
            return NULL;
        }

        buffer[length] = '\0';
        t->tokens[i].score = score;
        t->tokens[i].entry = buffer;
    }

    fclose(file);

    // Load prompt templates
    t->prompt = template_create(in_file, 0, enable_thinking);
    t->system = template_create(in_file, 1, enable_thinking);

    fprintf(stderr, "[Tokenizer] bos_id=%d\n", t->bos_id);
    fprintf(stderr, "[Tokenizer] eos_id=%d\n", t->eos_id);
    fprintf(stderr, "[Tokenizer] vocab_size=%d\n", t->vocab_size);
    fprintf(stderr, "[Tokenizer] max_token_length=%d\n", t->max_token_length);
    fprintf(stderr, "[Tokenizer] prompt\n%s\n", t->prompt->data);
    fprintf(stderr, "[Tokenizer] system\n%s\n", t->system->data);

    return t;
}

void tokenizer_free(Tokenizer* t) {
    if (!t) {
        return;
    }
    for (int i = 0; i < t->vocab_size; i++) {
        free(t->tokens[i].entry);
    }
    free(t->tokens);
    template_free(t->prompt);
    template_free(t->system);
    free(t);
}

/** @} */

/**
 * @section Tokenizer: Token Mapping
 * @{
 */

char* tokenizer_id_to_token(Tokenizer* t, int id) {
    if (!t || id < 0 || id >= t->vocab_size) {
        fprintf(stderr, "[tokenizer_id_to_token] ERROR: Invalid id! %d\n", id);
        return NULL;
    }
    return t->tokens[id].entry;
}

int tokenizer_token_to_id(Tokenizer* t, const char* token) {
    if (!t || !t->tokens || !token) {
        fprintf(stderr, "[tokenizer_token_to_id] ERROR: Invalid token! %s\n", token);
        return -1;
    }

    // find a match for str in vocab, return its index or -1 if not found
    for (int i = 0; i < t->vocab_size; i++) {
        if (!t->tokens[i].entry) {
            fprintf(
                stderr,
                "[tokenizer_token_to_id] Error: Malformed entry! "
                "i=%d, tokens=%p, entry=%s, score=%f\n",
                i,
                t->tokens[i],
                t->tokens[i].entry,
                t->tokens[i].score
            );
            return -1;
        }

        if (0 == strcmp(token, t->tokens[i].entry)) {
            return i;
        }
    }

    return -1;
}

/** @} */

/**
 * @section Tokenizer: Encoder
 * @{
 */

static int tokenizer_find_special_token(Tokenizer* t, char* start, char* out) {
    for (int k = 0; start[k] && k < t->max_token_length; ++k) {
        if (start[k] == '>') {
            int n_bytes = k + 1; // number of bytes consumed
            strncpy(out, start, n_bytes);
            out[n_bytes] = '\0';
            return n_bytes;
        }
    }
    return 0;
}

static int tokenizer_find_token_ids(Tokenizer* t, char* start, int* out) {
    int n_ids = 0;
    char* bytes = start;

    while (*bytes) {
        int id = -1;
        char token[t->max_token_length];
        token[0] = '\0';

        if (*bytes == '<') {
            int consumed = tokenizer_find_special_token(t, bytes, token);
            if (consumed > 0) {
                id = tokenizer_token_to_id(t, token);
                if (id != -1) {
                    bytes += consumed;
                }
            }
        }

        if (id == -1) {
            token[0] = *bytes++;
            token[1] = '\0';
            id = tokenizer_token_to_id(t, token);
        }

        if (id != -1) {
            out[(n_ids)++] = id;
        } else {
            fprintf(stderr, "Warning: Unknown character `%c` (codepoint %d)\n", token[0], token[0]);
        }
    }

    return n_ids;
}

static int tokenizer_find_best_merge(
    Tokenizer* t, int* ids, int n_ids, char* buf, int* out_id, int* out_index
) {
    float best_score = -1e10f;
    int best_id = -1;
    int best_idx = -1;

    for (int i = 0; i < n_ids - 1; ++i) {
        snprintf(
            buf,
            t->max_token_length * 2 + 1,
            "%s%s",
            t->tokens[ids[i]].entry,
            t->tokens[ids[i + 1]].entry
        );

        int merged_id = tokenizer_token_to_id(t, buf);
        if (merged_id != -1 && t->tokens[merged_id].score > best_score) {
            best_score = t->tokens[merged_id].score;
            best_id = merged_id;
            best_idx = i;
        }
    }

    if (best_idx != -1) {
        *out_id = best_id;
        *out_index = best_idx;
        return 1; // match found
    }

    return 0; // no match found
}

void tokenizer_merge_token_ids(Tokenizer* t, int* ids, int* n_ids) {
    char* buffer = malloc(t->max_token_length * 2 + 1);
    while (1) {
        int best_id, best_idx;
        if (!tokenizer_find_best_merge(t, ids, *n_ids, buffer, &best_id, &best_idx)) {
            break;
        }

        ids[best_idx] = best_id;
        memmove(&ids[best_idx + 1], &ids[best_idx + 2], (*n_ids - best_idx - 2) * sizeof(int));
        (*n_ids)--;
    }
    free(buffer);
}

void tokenizer_encode(Tokenizer* t, char* text, int* ids, int* n_ids) {
    // Initial tokenization
    *n_ids = tokenizer_find_token_ids(t, text, ids);
    tokenizer_merge_token_ids(t, ids, n_ids);
}

/** @} */

/**
 * @section Sampler
 * The Sampler, which takes logits and returns a sampled token
 * sampling can be done in a few ways: greedy argmax, sampling, top-p sampling
 * @{
 */

/**
 * @brief Probability-Index pair for top-p filtering and sampling.
 */
typedef struct Probability {
    float sample; // probability mass
    int index; // token index
} Probability;

typedef struct Sampler {
    Probability* probs; // buffer used in top-p sampling
    unsigned long long seed; // current rng state
    float temperature; // entropy scale
    float top_p; // top probability
    int vocab_size; // embed dim
} Sampler;

Sampler* sampler_create(int vocab_size, float temperature, float top_p, unsigned long long seed) {
    Sampler* s = calloc(1, sizeof(Sampler));
    if (!s) {
        return NULL;
    }

    // buffer only used with nucleus sampling; may not need but it's ~small
    s->probs = calloc(vocab_size, sizeof(Probability));
    if (!s->probs) {
        free(s);
        return NULL;
    }

    // clamp to [ε, 1.0f], where ε > 0
    const float epsilon = 1e-6f;
    if (top_p > 1.0f || isnan(top_p) || 1 == isinf(top_p)) {
        s->top_p = 1.0f; // upper limit
    } else if (top_p < epsilon || -1 == isinf(top_p)) {
        s->top_p = epsilon; // mitigate low probs
    } else {
        s->top_p = top_p;
    }

    // clamp to [ε, +/- 1.0f], where ε > 0
    if (isnan(temperature) || 1 == isinf(temperature)) {
        s->temperature = 1.0f; // sane default
    } else if (temperature < epsilon || -1 == isinf(temperature)) {
        s->temperature = epsilon; // mitigate division by zero
    } else {
        s->temperature = temperature;
    }

    s->vocab_size = vocab_size;
    s->seed = seed;

    fprintf(stderr, "[Sampler] vocab_size=%d\n", s->vocab_size);
    fprintf(stderr, "[Sampler] temperature=%f\n", s->temperature);
    fprintf(stderr, "[Sampler] top_p=%f\n", s->top_p);
    fprintf(stderr, "[Sampler] seed=%lu\n", s->seed);

    return s;
}

void sampler_free(Sampler* s) {
    if (s) {
        if (s->probs) {
            free(s->probs);
        }
        free(s);
    }
}

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
        mass += s->probs[i].sample;
        if (mass > s->top_p) {
            id = i;
            break;
        }
    }

    if (out_mass) {
        // Heal the sampled distribution
        const float epsilon = 1e-3f; // absolute threshold
        if (mass < epsilon) {
            for (int i = 0; i <= id; i++) {
                mass += s->probs[i].sample; // add weight to the distribution
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
    return dist[n - 1].index; // fallback
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
        s->probs[i].index = i;
        s->probs[i].sample = samples[i];
    }

    // Sort descending by probability
    qsort(s->probs, s->vocab_size, sizeof(Probability), sampler_cmp_dist);

    // Truncate where cumulative probability exceeds top_p
    float mass = 0.0f;
    int id = sampler_mass_index(s, &mass);
    // Sample from truncated distribution
    return sampler_cdf_index(s->probs, id, coin, mass);
}

/**
 * xorshift rng: generate next step in sequence [0, 2^64).
 * @ref https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
 */
unsigned int random_u32(unsigned long long* state) {
    *state ^= *state >> 12; // shift right, then flip
    *state ^= *state << 25; // shift left, then flip
    *state ^= *state >> 27; // shift right, then flip
    return (*state * 0x2545F4914F6CDD1Dull) >> 32; // scale, then drop 32-bits
}

/**
 * xorshift rng: normalize rng state [0, 1).
 */
float random_f32(unsigned long long* state) {
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler* sampler, float* logits) {
    /// @todo Apply temperature annealing for long seq lengths
    // apply the temperature to the logits
    for (int q = 0; q < sampler->vocab_size; q++) {
        logits[q] /= sampler->temperature; // scale
    }

    // apply softmax to the logits to get the samples for next token
    softmax(logits, sampler->vocab_size); // normalize
    // create a source of entropy for sampling
    float coin = random_f32(&sampler->seed); // flip a coin
    // top-p (nucleus) sampling, clamping the least likely tokens to zero
    return sampler_top_p(sampler, logits, coin);
}

/** @} */

/**
 * @section CLI Options
 */

typedef enum Thinking {
    THINKING_OFF,
    THINKING_ON,
} Thinking;

typedef struct Options {
    char* prompt; // optional (used in completions)
    char* system_prompt; // optional (used in chat completions)
    char* path; // required (e.g., model.bin)
    char* mode; // "completion" or "chat"
    unsigned long long seed; // seed rng with time by default
    Thinking thinking; // 1 enables thinking
    int seq_len; // max context length
    float temperature; // 0.0f = deterministic and 1.0f = creative
    float top_p; // nucleus sampling, 1.0 = off
} Options;

Options options_init(void) {
    // set default parameters
    return (Options) {
        .prompt = NULL,
        .system_prompt = NULL,
        .path = NULL,
        .mode = "chat",
        .seed = (unsigned long long) time(NULL),
        .temperature = 1.0f,
        .top_p = 0.9f,
        .thinking = THINKING_OFF,
        .seq_len = 0,
    };
}

static void options_usage(void) {
    fprintf(stderr, "Usage:   runq <checkpoint> [options]\n");
    fprintf(stderr, "Example: runq Qwen3-4B.bin -r 1\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>  top-p (nucleus sampling), default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -c <int>    context window size, 0 (default) = max_seq_len\n");
    fprintf(stderr, "  -m <string> mode: completion|chat, default: chat\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -y <string> system prompt (chat mode)\n");
    fprintf(stderr, "  -r <int>    reasoning mode: 0=off, 1=thinking\n");
}

int options_parse(Options* o, int argc, char** argv) {
    if (argc < 2) {
        options_usage();
        return 1;
    }

    o->path = argv[1];

    for (int i = 2; i < argc; i += 2) {
        if (i + 1 >= argc || argv[i][0] != '-' || strlen(argv[i]) != 2) {
            options_usage();
            return -1;
        }
        char flag = argv[i][1];
        char* arg = argv[i + 1];

        switch (flag) {
            case 't':
                o->temperature = atof(arg);
                break;
            case 'p':
                o->top_p = atof(arg);
                break;
            case 's':
                {
                    int seed = abs(atoi(arg));
                    if (seed) {
                        o->seed = (unsigned long long) seed;
                    }
                }
                break;
            case 'c':
                int seq_len = abs(atoi(arg));
                o->seq_len = (seq_len > MAX_SEQ_LEN) ? MAX_SEQ_LEN : seq_len;
                break;
            case 'i':
                o->prompt = arg;
                break;
            case 'y':
                o->system_prompt = arg;
                break;
            case 'm':
                {
                    o->mode = arg;
                }
                break;
            case 'r':
                o->thinking = atoi(arg) ? THINKING_ON : THINKING_OFF;
                break;
            default:
                options_usage();
                return 1;
        }
    }

    return 0;
}

/** @} */

/**
 * @section Transformer Model
 * @{
 */

typedef struct Qwen {
    Transformer* model;
    Tokenizer* tokenizer;
    Sampler* sampler;
} Qwen;

Qwen* qwen_create(Options* o) {
    Qwen* q = calloc(1, sizeof(Qwen));
    if (!q) {
        return NULL;
    }

    // build the Tokenizer via the tokenizer .bin file
    q->tokenizer = tokenizer_create(o->path, o->thinking);
    if (!q->tokenizer) {
        goto tokenizer_failed;
    }

    // build the Transformer via the model .bin file
    q->model = transformer_create(o->path, o->seq_len);
    if (!q->model) {
        goto model_failed;
    }

    // build the Sampler
    q->sampler = sampler_create(q->model->params.vocab_size, o->temperature, o->top_p, o->seed);
    if (!q->sampler) {
        goto sampler_failed;
    }

    return q;

sampler_failed:
    transformer_free(q->model);
model_failed:
    tokenizer_free(q->tokenizer);
tokenizer_failed:
    free(q);
    return NULL;
}

void qwen_free(Qwen* q) {
    if (q) {
        if (q->model) {
            transformer_free(q->model);
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

/**
 * @section Completions
 * @{
 */

/**
 * @brief Autoregressive text completion.
 *
 * Generates a continuation for the given input `prompt` using
 * the transformer, tokenizer, and sampler provided.
 *
 * The prompt is first encoded into token IDs. The transformer
 * consumes the prompt tokens sequentially (teacher forcing),
 * after which sampling begins for each subsequent token until:
 *   - The model predicts BOS/EOS (sequence termination), or
 *   - The context length (seq_len) is reached.
 *
 * @param transformer Pointer to the Transformer model.
 * @param tokenizer   Pointer to the Tokenizer (handles encoding/decoding).
 * @param sampler     Pointer to the Sampler (controls temperature/top-p).
 * @param prompt      UTF-8 input string to complete.
 *
 * @note This function streams tokens directly to stdout.
 * @note Deterministic generation only occurs if temperature ~0
 *       and the RNG seed is fixed.
 */
void completion(Qwen* qwen, char* prompt) {
    fprintf(stderr, "[Completion]\n");

    // Validate prompt
    if (!prompt || strlen(prompt) == 0) {
        fprintf(stderr, "[Completion] Error: Missing prompt. Use -i 'string'.\n");
        exit(EXIT_FAILURE);
    }

    // Encode the prompt into token IDs
    int* ids = calloc(strlen(prompt) + 1, sizeof(int)); // +1 for null terminator
    if (!ids) {
        fprintf(stderr, "[Completion] Error: Failed to allocate memory for input ids.\n");
        exit(EXIT_FAILURE);
    }

    int n_ids = 0;
    tokenizer_encode(qwen->tokenizer, prompt, ids, &n_ids);
    if (n_ids < 1) {
        fprintf(stderr, "[Completion] Error: Failed to encode input prompt.\n");
        free(ids);
        exit(EXIT_FAILURE);
    }

    // Initialize autoregressive state
    int token = ids[0]; // first token from prompt
    int next = 0; // next token to be generated

    for (int pos = 0; pos < qwen->model->params.seq_len; pos++) {
        // Forward pass: compute logits for this position
        float* logits = forward(qwen->model, token, pos);

        // Teacher forcing for prompt, sampling for generation
        if (pos + 1 < n_ids) {
            next = ids[pos + 1]; // still consuming prompt
        } else {
            next = sample(qwen->sampler, logits); // now generating
        }

        // Decode and stream the current token
        printf("%s", tokenizer_id_to_token(qwen->tokenizer, token));
        fflush(stdout);

        // Stop if BOS/EOS encountered after prompt
        if (next == qwen->tokenizer->bos_id || next == qwen->tokenizer->eos_id) {
            break;
        }

        // Advance autoregressive state
        token = next;
    }

    printf("\n");
    free(ids);
}

/** @} */

/**
 * @section Chat Completions
 * @{
 */

void chat_input(const char* prompt, char* buffer, size_t bufsize) {
    printf("%s", prompt);
    if (fgets(buffer, bufsize, stdin)) {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n') {
            buffer[len - 1] = '\0'; // strip newline
        }
    }
}

bool chat_reset(int* user, int* pos) {
    if (!user || !pos) {
        return false;
    }
    printf("\n[ChatCompletion] Context window full! Resetting turn=%d, pos=%d.\n", *user, *pos);
    *user = 1;
    *pos = 0;
    return true;
}

void chat_completion(Qwen* qwen, char* system_prompt) {
    fprintf(stderr, "[ChatCompletion]\n");

    char prompt[MAX_SEQ_LEN];
    char template[MAX_SEQ_LEN];

    int user = 1; // user starts
    int u_id = 0; // user token id
    int n_ids = 0; // number of token ids
    int* ids = (int*) malloc(MAX_SEQ_LEN * sizeof(int)); // array of token ids

    // start the main loop
    int token = 0; // stores the current token to feed into the transformer
    int next = 0; // will store the next token in the sequence
    int pos = 0; // position in the sequence
    while (1) {
        // if context window is exceeded, clear it
        if (pos >= qwen->model->params.seq_len) {
            chat_reset(&user, &pos);
        }

        // when it is the user's turn to contribute tokens to the dialog...
        if (user) {
            // get the user prompt from stdin
            chat_input("\n> ", prompt, sizeof(prompt));
            // terminate if user enters a blank prompt
            if (!prompt[0]) {
                break;
            }

            // render user/system prompts into the Qwen3 prompt template schema
            if (pos == 0 && system_prompt) {
                sprintf(template, qwen->tokenizer->system->data, system_prompt, prompt);
            } else {
                sprintf(template, qwen->tokenizer->prompt->data, prompt);
            }

            // encode the rendered prompt into tokens
            tokenizer_encode(qwen->tokenizer, template, ids, &n_ids);
            u_id = 0; // reset the user index
            user = 0;
        }

        // determine the token to pass into the transformer next
        if (u_id < n_ids) {
            // if we are still processing the input prompt, force the next prompt token
            token = ids[u_id++];
        } else {
            // otherwise use the next token sampled from previous turn
            token = next;
        }

        // printf("|pos=%d token=%d '%s'|\n",pos,token,tokenizer->vocab[token]);

        // forward the transformer to get logits for the next token
        float* logits = forward(qwen->model, token, pos);
        next = sample(qwen->sampler, logits);
        pos++;

        // assistant is responding
        if (u_id >= n_ids) {
            if (token == qwen->tokenizer->bos_id || token == qwen->tokenizer->eos_id) {
                // EOS token ends the assistant turn
                printf("\n");
                user = 1;
            } else if (next != qwen->tokenizer->bos_id && next != qwen->tokenizer->eos_id) {
                printf("%s", tokenizer_id_to_token(qwen->tokenizer, next));
                fflush(stdout);
            }
        }
    }

    free(ids);
}

/** @} */

/**
 * @section main
 * @{
 */

int main(int argc, char* argv[]) {
    Options opts;
    if (options_parse(&opts, argc, argv) < 0) {
        return EXIT_FAILURE;
    }

    Qwen* qwen = qwen_create(&opts);
    if (!qwen) {
        fprintf(stderr, "[Error] Failed to initialize Qwen\n");
        return EXIT_FAILURE;
    }

    if (strcmp(opts.mode, "completion") == 0) {
        completion(qwen, opts.prompt);
    } else if (strcmp(opts.mode, "chat") == 0) {
        chat_completion(qwen, opts.system_prompt);
    } else {
        fprintf(stderr, "Unknown mode: %s\n", opts.mode);
    }

    qwen_free(qwen);
    return EXIT_SUCCESS;
}

/** @} */
