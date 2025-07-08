/**
 * @file src/forward.c
 * @brief Forward pass for Transformer checkpoints
 */

#include "forward.h"
#include <math.h>
#include <string.h>

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
    State* s = &t->state;

    int head_dim = p->head_dim;
    int kv_mul = p->n_heads / p->n_kv_heads;
    int kv_dim = p->n_kv_heads * head_dim;
    uint64_t loff = (uint64_t) l * p->seq_len * kv_dim;

    for (int h = 0; h < p->n_heads; h++) {
        float* q = s->q + h * head_dim;
        float* att_scores = s->att_scores + h * p->seq_len;
        float* head_out = s->x_rms_norm + h * head_dim;

// Compute attention scores
#pragma omp parallel for
        for (int i = 0; i <= pos; i++) {
            float* k = s->k_cache + loff + i * kv_dim + (h / kv_mul) * head_dim;
            float score = 0.0f;
            for (int j = 0; j < head_dim; j++) {
                score += q[j] * k[j];
            }

            att_scores[i] = score / sqrtf((float) head_dim);
        }

        // Normalize scores to get attention weights
        softmax(att_scores, pos + 1);

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
                    tmp[j] += att_scores[i] * v[j];
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
    State* s = &t->state;

    int kv_dim = p->n_kv_heads * p->head_dim;
    // int kv_mul = p->n_heads / p->n_kv_heads; // multi-query attention (currently unused)
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
