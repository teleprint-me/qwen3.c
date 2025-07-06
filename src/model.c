/// @file src/model.c
#include "model.h"
#include <stdlib.h>
#include <stdio.h>

State* state_create(Params* p) {
    State* s = calloc(1, sizeof(State));
    if (!s) {
        fprintf(stderr, "state allocation failed!\n");
        return NULL;
    }

    const int hidden_dim = p->hidden_dim;
    const int projection = p->n_heads * p->head_dim; // IO Features
    const int kv_dim = p->n_kv_heads * p->head_dim;
    const uint64_t cache_len = (uint64_t) p->n_layers * p->seq_len * kv_dim;

    assert(projection % GS == 0 && "projection must be divisible by GS");
    assert(hidden_dim % GS == 0 && "hidden_dim must be divisible by GS");

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

    s->qx.q = calloc(projection, sizeof(int8_t));
    s->qx.s = calloc(projection / GS, sizeof(float));

    s->qh.q = calloc(hidden_dim, sizeof(int8_t));
    s->qh.s = calloc(hidden_dim / GS, sizeof(float));

    // Check for allocation failures
    if (!s->x || !s->r || !s->att_out || !s->q || !s->att || !s->logits || !s->k_cache
        || !s->v_cache || !s->mlp_in || !s->mlp_gate || !s->qx.q || !s->qx.s || !s->qh.q
        || !s->qh.s) {
        fprintf(stderr, "state_create: malloc failed!\n");
        exit(EXIT_FAILURE);
    }

    size_t total_bytes =
        p->dim * 3 * sizeof(float) + // x, r, att_out
        projection * (2 * sizeof(float) + sizeof(int8_t)) + // q, r, qx.q
        (projection / GS) * sizeof(float) + // qx.s
        hidden_dim * (2 * sizeof(float) + sizeof(int8_t)) + // mlp, mlp_gate, qh.q
        (hidden_dim / GS) * sizeof(float) + // qh.s
        p->n_heads * p->seq_len * sizeof(float) + // att
        p->vocab_size * sizeof(float) + // logits
        2 * cache_len * sizeof(float); // kv_cache
    fprintf(stderr, "State memory use: %.2f MB\n", total_bytes / (1024.0 * 1024.0));

    return s;
}

void state_free(State* s) {
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
    free(s);
}
