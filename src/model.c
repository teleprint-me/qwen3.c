/// @file src/model.c
#include "model.h"
#include <stdbool.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <string.h>
#include <stdio.h>

/**
 * @section Model State
 */

State* state_create(Params* p) {
    if (!p) {
        return NULL;
    }

    State* s = calloc(1, sizeof(State));
    if (!s) {
        fprintf(stderr, "state_create: allocation failed!\n");
        return NULL;
    }

    const int hidden_dim = p->hidden_dim;
    const int projection = p->n_heads * p->head_dim; // IO Features
    const int kv_dim = p->n_kv_heads * p->head_dim;
    const uint64_t cache_len = (uint64_t) p->n_layers * p->seq_len * kv_dim;

    assert(0 == projection % GS && "projection must be divisible by GS");
    assert(0 == hidden_dim % GS && "hidden_dim must be divisible by GS");
    assert(0 != cache_len && "Empty cache size");

    // Residual stream and attention output
    s->x = calloc(p->dim, sizeof(float)); // persistent
    s->r = calloc(projection, sizeof(float)); // scratch for norm/project
    s->att_out = calloc(p->dim, sizeof(float)); // attention output (before residual)

    // Attention workspace
    s->q = calloc(projection, sizeof(float));
    s->k = NULL; // s->k and s->v are aliases into slices of k_cache and v_cache
    s->v = NULL; // They point to the current time step within layer 'l'
    s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));

    // Key/value memory (shared memory with KV)
    s->k_cache = calloc(cache_len, sizeof(float));
    s->v_cache = calloc(cache_len, sizeof(float));

    // MLP
    s->mlp_in = calloc(hidden_dim, sizeof(float));
    s->mlp_gate = calloc(hidden_dim, sizeof(float));

    // qx.q stores int8_t quantized values of r (projection dim)
    s->qx.q = calloc(projection, sizeof(int8_t));
    // qx.s stores per-group scale factors (projection / GS)
    s->qx.s = calloc(projection / GS, sizeof(float));

    s->qh.q = calloc(hidden_dim, sizeof(int8_t));
    s->qh.s = calloc(hidden_dim / GS, sizeof(float));

    // Check for allocation failures
    if (!s->x || !s->r || !s->att_out || !s->q || !s->att || !s->logits || !s->k_cache
        || !s->v_cache || !s->mlp_in || !s->mlp_gate || !s->qx.q || !s->qx.s || !s->qh.q
        || !s->qh.s) {
        fprintf(stderr, "state_create: allocation failed!\n");
        return NULL;
    }

    size_t total_bytes = p->dim * 3 * sizeof(float) + // x, r, att_out
                         projection * (2 * sizeof(float) + sizeof(int8_t)) + // q, r, qx.q
                         (projection / GS) * sizeof(float) + // qx.s
                         hidden_dim * (2 * sizeof(float) + sizeof(int8_t)) + // mlp, mlp_gate, qh.q
                         (hidden_dim / GS) * sizeof(float) + // qh.s
                         p->n_heads * p->seq_len * sizeof(float) + // att
                         p->vocab_size * sizeof(float) + // logits
                         2 * cache_len * sizeof(float); // kv_cache
    fprintf(stderr, "state_create: allocated %.2f MB\n", total_bytes / (1024.0 * 1024.0));

    return s;
}

void state_free(State* s) {
    if (!s) {
        return;
    }

    // Residual stream and attention output
    free(s->x);
    free(s->r);
    free(s->att_out);

    // Attention workspace
    free(s->q);
    free(s->att);
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

    // Free the struct
    free(s);
}

/** @} */

/**
 * @section Model Weights
 */

Weights* weights_create(Params* p, void* stream) {
    if (!p || !stream) {
        return NULL;
    }

    Weights* w = calloc(1, sizeof(Weights));
    if (!w) {
        return NULL;
    }

    /**
     * FP32 Weights (RMSNorms + LayerNorms)
     * These are read directly from the stream without allocation.
     * Layout order must match export script.
     */
    float* weights = (float*) stream;

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
     * q8_tensor allocates memory for Q8Tensor and updates the stream pointer.
     */
    stream = (void*) weights;

    // Token embeddings (quantized + dequantized)
    w->qe = q8_tensor(&stream, 1, p->vocab_size * p->dim); // allocates internally
    w->fe = calloc(p->vocab_size * p->dim, sizeof(float)); // explicit malloc (must be freed)
    if (!w->fe) {
        free(w);
        return NULL;
    }

    q8_dequantize(w->qe, w->fe, p->vocab_size * p->dim);

    /**
     * Attention weights
     * All tensors are shaped [n_layers, dim * out_features] for consistent layout.
     * Matmul kernels must handle reshaping internally.
     */
    const int projection = p->n_heads * p->head_dim;
    const int kv_dim = p->n_kv_heads * p->head_dim;

    w->wq = q8_tensor(&stream, p->n_layers, p->dim * projection);
    w->wk = q8_tensor(&stream, p->n_layers, p->dim * kv_dim);
    w->wv = q8_tensor(&stream, p->n_layers, p->dim * kv_dim);
    w->wo = q8_tensor(&stream, p->n_layers, projection * p->dim); // [proj, dim] format

    /**
     * Feed-forward weights
     * All three MLP branches use [hidden_dim × dim] layout in export
     */
    const int hidden_dim = p->hidden_dim;

    w->w1 = q8_tensor(&stream, p->n_layers, p->dim * hidden_dim); // w1(x)
    w->w2 = q8_tensor(&stream, p->n_layers, hidden_dim * p->dim); // w2(silu ⊙ w3(x))
    w->w3 = q8_tensor(&stream, p->n_layers, p->dim * hidden_dim); // w3(x)

    /**
     * Output classifier
     * If shared_classifier is true, reuse token embedding matrix
     * (tied weights). Otherwise, allocate separate output projection.
     */
    w->cls = p->shared_classifier ? w->qe : q8_tensor(&stream, 1, p->dim * p->vocab_size);

    return w;
}

void weights_free(Params* p, Weights* w) {
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

    // Free the struct
    free(w);
}

/** @} */

/**
 * @section Transformer Model
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

bool model_read_params(Transformer* t, int override_seq_len) {
    if (!t) {
        return false;
    }

    memcpy(&t->params, t->model, sizeof(Params));
    if (0x7177656E != t->params.magic_number || 1 != t->params.version) {
        return false;
    }

    if (override_seq_len && override_seq_len <= t->params.seq_len) {
        t->params.seq_len = override_seq_len;
    }

    GS = t->params.group_size;
    return true;
}

Transformer* transformer_create(const char* path, int override_seq_len) {
    if (!path) {
        return NULL;
    }

    Transformer* t = calloc(1, sizeof(Transformer));
    if (!t) {
        return NULL;
    }

    // Read and map the model file
    if (!model_read_checkpoint(t, path)) {
        return NULL;
    }

    if (!model_read_params(t, override_seq_len)) {
        return NULL;
    }

    return t;
}

/** @} */
