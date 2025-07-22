/// @file src/model.c
#include "model.h"
#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

/**
 * @defgroup Private Interface
 * @{
 */

/**
 * @section Model Checkpoint
 */

bool model_file_mmap(Model* m, const char* path) {
    FILE* file = fopen(path, "rb");
    if (!file) {
        goto open_failure;
    }

    if (-1 == fseek(file, 0, SEEK_END)) {
        goto read_failure;
    }

    m->size = ftell(file);
    if (-1 == m->size) {
        goto read_failure;
    }

    m->data = mmap(NULL, m->size, PROT_READ, MAP_PRIVATE, fileno(file), 0);
    if (!m->data || m->data == MAP_FAILED) {
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
///       Otherwise the repeated computations just compound one another over
///       time.
bool model_params_mmap(Model* m, int override_seq_len) {
    if (!m) {
        return false;
    }

    memcpy(&m->params, m->data, sizeof(ModelParams));
    if (QWEN_MAGIC != m->params.magic || QWEN_VERSION != m->params.version) {
        return false;
    }

    assert(
        m->params.block_size % 2
        && "[Params] block_size must be evenly divisible."
    );

    if (override_seq_len > 0 && override_seq_len <= m->params.seq_len) {
        m->params.seq_len = override_seq_len;
    }

    const int padding = 256;
    m->data = ((char*) m->data) + padding;  // add alignment to weights

    fprintf(stderr, "[Params] magic=%x\n", m->params.magic);
    fprintf(stderr, "[Params] version=%d\n", m->params.version);
    fprintf(stderr, "[Params] hidden_size=%d\n", m->params.dim);
    fprintf(stderr, "[Params] intermediate_size=%d\n", m->params.hidden_dim);
    fprintf(stderr, "[Params] num_hidden_layers=%d\n", m->params.n_layers);
    fprintf(stderr, "[Params] num_attention_heads=%d\n", m->params.n_heads);
    fprintf(stderr, "[Params] num_kv_heads=%d\n", m->params.n_kv_heads);
    fprintf(stderr, "[Params] vocab_size=%d\n", m->params.vocab_size);
    fprintf(stderr, "[Params] seq_len=%d\n", m->params.seq_len);
    fprintf(stderr, "[Params] head_dim=%d\n", m->params.head_dim);
    fprintf(
        stderr, "[Params] shared_classifier=%d\n", m->params.shared_classifier
    );
    fprintf(stderr, "[Params] block_size=%d\n", m->params.block_size);

    return true;
}

/**
 * @section Model Tensors
 * @{
 */

/**
 * Parses an array of Q8_0 quantized tensors from flat memory.
 *
 * Each tensor is stored as:
 *   - [int8_t[size]]: flattened weight
 *   - [float[size / GS]]: scale factors (1 per block)
 *
 * The input `*X` is expected to point to the start of this data.
 * On return, `*X` will be advanced past the parsed region.
 *
 * @param b Pointer to raw memory buffer (usually mapped model)
 * @param n Number of tensors to parse
 * @param size Number of int8_t elements in each tensor (must be divisible by
 * GS)
 * @return Q8Tensor array with n entries (caller must free)
 */
Q8Tensor* q8_tensor_mmap(Model* m, int tensors, int integers, int block_size) {
    if (!m || !m->data || tensors <= 0 || integers <= 0 || block_size <= 0) {
        return NULL;
    }

    Q8Tensor* qt = calloc(tensors, sizeof(Q8Tensor));
    if (!qt) {
        return NULL;
    }

    // Buffer must be read linearly
    for (int i = 0; i < tensors; i++) {
        // map q8 values
        qt[i].q = (int8_t*) m->data;
        m->data = (int8_t*) m->data + integers;

        // map scalars
        qt[i].s = (float*) m->data;
        m->data = (float*) m->data + integers / block_size;
    }

    return qt;
}

/** @} */

/**
 * @section Model Weights
 */

/**
 * @brief Initialize and allocate quantized and fp32 weight tensors from
 * memory-mapped stream.
 *
 * This function assumes `stream` points to a contiguous memory-mapped model
 * checkpoint. All fp32 weights (e.g. RMSNorm parameters) are read first, then
 * the quantized tensors are constructed using `q8_tensor`, which allocates
 * memory internally and adjusts the stream pointer.
 *
 * @param t        Pointer to Transformer model.
 * @return true on success, or false on error.
 */
bool model_weights_mmap(Model* m) {
    if (!m || !m->data || 0 == m->size) {
        return false;
    }

    /**
     * FP32 Weights (RMSNorms + LayerNorms)
     * These are read directly from the stream without allocation.
     * Layout order must match export script.
     */
    ModelParams* p = &m->params;
    ModelWeights* w = &m->weights;
    float* weights = (float*) m->data;

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
     * q8_tensor_map allocates memory for Q8Tensor and updates the stream
     * pointer.
     */
    m->data = (void*) weights;

    // Token embeddings (quantized + dequantized; allocates internally)
    w->qe = q8_tensor_mmap(m, 1, p->vocab_size * p->dim, p->block_size);
    // explicit malloc (must be freed)
    w->fe = calloc(p->vocab_size * p->dim, sizeof(float));
    if (!w->fe) {
        return false;
    }

    q8_dequantize(w->qe, w->fe, p->vocab_size * p->dim, p->block_size);

    /**
     * Attention weights
     * All tensors are shaped [n_layers, dim * out_features] for consistent
     * layout. Matmul kernels must handle reshaping internally.
     */
    const int proj_dim = p->n_heads * p->head_dim;
    const int kv_dim = p->n_kv_heads * p->head_dim;

    w->wq = q8_tensor_mmap(m, p->n_layers, p->dim * proj_dim, p->block_size);
    w->wk = q8_tensor_mmap(m, p->n_layers, p->dim * kv_dim, p->block_size);
    w->wv = q8_tensor_mmap(m, p->n_layers, p->dim * kv_dim, p->block_size);
    w->wo = q8_tensor_mmap(m, p->n_layers, proj_dim * p->dim, p->block_size);

    /**
     * Feed-forward weights
     * All three MLP branches use [hidden_dim × dim] layout in export
     */
    const int hidden_dim = p->hidden_dim;

    w->w1 = q8_tensor_mmap(
        m, p->n_layers, p->dim * hidden_dim, p->block_size
    );  // w1(x)
    w->w2 = q8_tensor_mmap(
        m, p->n_layers, hidden_dim * p->dim, p->block_size
    );  // w2(silu ⊙ w3(x))
    w->w3 = q8_tensor_mmap(
        m, p->n_layers, p->dim * hidden_dim, p->block_size
    );  // w3(x)

    /**
     * Output classifier
     * If shared_classifier is true, reuse token embedding matrix
     * (tied weights). Otherwise, allocate separate output proj_dim.
     */
    w->cls = p->shared_classifier
                 ? w->qe
                 : q8_tensor_mmap(m, 1, p->dim * p->vocab_size, p->block_size);

    size_t total_bytes =
        // FP32 weights
        p->n_layers * p->dim * 2 * sizeof(float) +  // att + ffn
        p->dim * sizeof(float) +  // out
        p->n_layers * p->head_dim * 2 * sizeof(float) +  // q and k

        // Token Embeddings
        p->vocab_size * p->dim * sizeof(int8_t) +  // qe.q
        (p->vocab_size * p->dim / p->block_size) * sizeof(float) +  // qe.s
        p->vocab_size * p->dim * sizeof(float) +  // fe

        // Attention weights
        2 * p->n_layers * p->dim * proj_dim * sizeof(int8_t) +  // wq, wo (q)
        2 * p->n_layers * (p->dim * proj_dim / p->block_size) * sizeof(float)
        +  // wq, wo (s)
        2 * p->n_layers * p->dim * kv_dim * sizeof(int8_t) +  // wk, wv (q)
        2 * p->n_layers * (p->dim * kv_dim / p->block_size) * sizeof(float)
        +  // wk, wv (s)

        // Feedforward weights
        3 * p->n_layers * p->dim * hidden_dim * sizeof(int8_t)
        +  // w1, w2, w3 (q)
        3 * p->n_layers * (p->dim * hidden_dim / p->block_size)
            * sizeof(float);  // w1, w2, w3 (s)

    if (!p->shared_classifier) {
        total_bytes += p->dim * p->vocab_size * sizeof(int8_t);  // cls.q
        total_bytes += (p->dim * p->vocab_size / p->block_size)
                       * sizeof(float);  // cls.s
    }

    fprintf(
        stderr, "[Weights] Allocated %.2f MB\n", total_bytes / (1024.0 * 1024.0)
    );

    return true;
}

void model_weights_free(Model* m) {
    if (!m) {
        return;
    }

    ModelParams* p = &m->params;
    ModelWeights* w = &m->weights;
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
 * @section Model State
 */

bool model_state_create(Model* m) {
    if (!m) {
        return false;
    }

    ModelParams* p = &m->params;
    ForwardState* s = &m->state;
    if (!p || !s) {
        return false;
    }

    const int proj_dim = p->n_heads * p->head_dim;  // IO Features
    const int kv_dim = p->n_kv_heads * p->head_dim;
    const uint64_t cache_len = (uint64_t) p->n_layers * p->seq_len * kv_dim;

    assert(
        0 == proj_dim % p->block_size
        && "[ForwardState] proj_dim must be divisible by block_size"
    );
    assert(
        0 == p->hidden_dim % p->block_size
        && "[ForwardState] hidden_dim must be divisible by block_size"
    );
    assert(0 != cache_len && "[ForwardState] cache_len must be greater than 0");

    // Residual stream and attention output
    s->x = calloc(p->dim, sizeof(float));  // persistent
    // norm/project
    s->x_rms_norm = calloc(proj_dim, sizeof(float));  // scratch

    // Attention workspace
    s->q = calloc(proj_dim, sizeof(float));
    s->k = NULL;  // s->k and s->v are aliases into slices of k_cache and
                  // v_cache
    s->v = NULL;  // They point to the current time step within layer 'l'
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
        .s = calloc(proj_dim / p->block_size, sizeof(float)),
    };

    s->qh = (Q8Tensor) {
        .q = calloc(p->hidden_dim, sizeof(int8_t)),
        .s = calloc(p->hidden_dim / p->block_size, sizeof(float)),
    };

    // Check for allocation failures
    if (!s->x || !s->x_rms_norm || !s->q || !s->scores || !s->logits
        || !s->k_cache || !s->v_cache || !s->mlp_in || !s->mlp_gate || !s->qx.q
        || !s->qx.s || !s->qh.q || !s->qh.s) {
        fprintf(stderr, "[ForwardState] Allocation failed!\n");
        return false;
    }

    size_t total_bytes = p->dim * 3 * sizeof(float) +  // x, x_rms_norm
                         proj_dim * (2 * sizeof(float) + sizeof(int8_t))
                         +  // q, x_rms_norm, qx.q
                         (proj_dim / p->block_size) * sizeof(float) +  // qx.s
                         p->hidden_dim * (2 * sizeof(float) + sizeof(int8_t))
                         +  // mlp, mlp_gate, qh.q
                         (p->hidden_dim / p->block_size) * sizeof(float)
                         +  // qh.s
                         p->n_heads * p->seq_len * sizeof(float) +  // scores
                         p->vocab_size * sizeof(float) +  // logits
                         2 * cache_len * sizeof(float);  // kv_cache

    fprintf(
        stderr,
        "[ForwardState] Allocated %.2f MB\n",
        total_bytes / (1024.0 * 1024.0)
    );

    return true;
}

void model_state_free(Model* m) {
    if (!m) {
        return;
    }

    ForwardState* s = &m->state;
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
 * @defgroup Public Interface
 * @{
 */

/**
 * @section Transformer Model
 */

Model* model_create(const char* path, int override_seq_len) {
    if (!path) {
        return NULL;
    }

    Model* m = calloc(1, sizeof(Model));
    if (!m) {
        goto malloc_failure;
    }

    if (!model_file_mmap(m, path)) {
        goto read_failure;
    }

    void* base = m->data;  // save the pointer
    if (!model_params_mmap(m, override_seq_len)) {
        goto read_failure;
    }

    if (!model_weights_mmap(m)) {
        goto read_failure;
    }

    if (!model_state_create(m)) {
        goto state_failure;
    }

    // Success: Return control flow
    m->data = base;  // rewind to start
    return m;

    // Failure: Break control flow
state_failure:
    model_weights_free(m);
read_failure:
    free(m);
malloc_failure:
    return NULL;
}

void model_free(Model* m) {
    if (!m) {
        return;
    }

    model_state_free(m);
    model_weights_free(m);
    munmap(m->data, m->size);
    free(m);
}

/** @} */
