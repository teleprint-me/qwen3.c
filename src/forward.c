/// @file src/forward.c
#include "q8.h"
#include "checkpoint.h"

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
        float* att = s->att + h * p->seq_len;
        float* r = s->r + h * head_dim;

// Compute attention scores
#pragma omp parallel for
        for (int i = 0; i <= pos; i++) {
            float* k = s->k_cache + loff + i * kv_dim + (h / kv_mul) * head_dim;
            float score = 0.0f;
            for (int j = 0; j < head_dim; j++) {
                score += q[j] * k[j];
            }

            att[i] = score / sqrtf((float) head_dim);
        }

        // Normalize scores to get attention weights
        softmax(att, pos + 1);

        // Initialize output vector for this head
        memset(r, 0, sizeof(float) * head_dim);

// Accumulate weighted values in thread-safe way
#pragma omp parallel
        {
            float tmp[head_dim];
            memset(tmp, 0, head_dim * sizeof(float));

#pragma omp for
            for (int i = 0; i <= pos; i++) {
                float* v = s->v_cache + loff + i * kv_dim + (h / kv_mul) * head_dim;
                float a = att[i];
                for (int j = 0; j < head_dim; j++) {
                    tmp[j] += a * v[j];
                }
            }

            // Reduce thread-local buffer into shared output
            for (int i = 0; i < head_dim; i++) {
#pragma omp atomic
                r[i] += tmp[i];
            }
        }
    }
}

float* forward(Transformer* t, int token, int pos) {
    Params* p = &t->params;
    Weights* w = &t->weights;
    State* s = &t->state;

    int dim = p->dim;
    int kv_dim = p->n_kv_heads * p->head_dim;
    int kv_mul = p->n_heads / p->n_kv_heads; // grouped-query attention
    int hidden_dim = p->hidden_dim;
    int proj_dim = p->n_heads * p->head_dim;

    // copy the token embedding into x
    memcpy(s->x, w->fe + token * dim, dim * sizeof(float));

    // forward all the layers
    for (int l = 0; l < p->n_layers; l++) {
        // KV cache layer offset
        uint64_t loff = l * (uint64_t) p->seq_len * kv_dim;

        // Save KV at this time step (pos) to cache
        s->k = s->k_cache + loff + pos * kv_dim;
        s->v = s->v_cache + loff + pos * kv_dim;

        // Normalize the input to attention.
        rmsnorm(s->r, s->x, w->att_rms_norm + l * dim, dim);

        // Quantize for matmul efficiency.
        q8_quantize(&s->qx, s->r, dim);

        // Compute Q, K, V for this timestep.
        matmul(s->q, &s->qx, w->wq + l, dim, proj_dim);
        matmul(s->k, &s->qx, w->wk + l, dim, kv_dim);
        matmul(s->v, &s->qx, w->wv + l, dim, kv_dim);

        float* gq = w->q_rms_norm + l * p->head_dim; // 128 floats
        float* gk = w->k_rms_norm + l * p->head_dim; // 128 floats

        /**
         * Q-RMSNorm + rotate each query head
         */
        for (int h = 0; h < p->n_heads; h++) {
            float* q = s->q + h * p->head_dim;

            rmsnorm(q, q, gq, p->head_dim);
            rotary(q, p->head_dim, pos);
        }

        /**
         * K-RMSNorm + rotate each key head
         */
        for (int h = 0; h < p->n_kv_heads; h++) {
            float* k = s->k + h * p->head_dim;

            rmsnorm(k, k, gk, p->head_dim);
            rotary(k, p->head_dim, pos);
        }

        /**
         * Multi-headed attention
         */
        attention(t, l, pos);

        // final matmul to get the output of the attention
        q8_quantize(&s->qx, s->r, proj_dim);
        matmul(s->att_proj, &s->qx, w->wo + l, proj_dim, dim);

        // residual connection back into x
        for (int i = 0; i < dim; i++) {
            s->x[i] += s->att_proj[i];
        }

        // ffn rmsnorm
        rmsnorm(s->r, s->x, w->ffn_rms_norm + l * dim, dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        q8_quantize(&s->qx, s->r, dim);
        matmul(s->mlp_in, &s->qx, w->w1 + l, dim, hidden_dim);
        matmul(s->mlp_gate, &s->qx, w->w3 + l, dim, hidden_dim);

        // SwiGLU non-linearity
        swiglu(s->mlp_in, s->mlp_gate, hidden_dim);

        // final matmul to get the output of the ffn
        q8_quantize(&s->qh, s->mlp_in, hidden_dim);
        matmul(s->r, &s->qh, w->w2 + l, hidden_dim, dim);

        // residual connection
        for (int i = 0; i < dim; i++) {
            s->x[i] += s->r[i];
        }
    }

    // final rmsnorm
    rmsnorm(s->x, s->x, w->out_rms_norm, dim);

    // classifier into logits
    q8_quantize(&s->qx, s->x, dim);
    matmul(s->logits, &s->qx, w->cls, dim, p->vocab_size);
    return s->logits;
}
