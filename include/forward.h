/// @file include/forward.h
/// @brief Forward pass for Transformer checkpoints
#ifndef QWEN_FORWARD_H
#define QWEN_FORWARD_H

#include "q8.h"
#include "checkpoint.h"

/**
 * @brief RMS normalization followed by elementwise scaling using a learned weight vector w.
 * @param out Output buffer (same size as x)
 * @param x Input vector to normalize
 * @param w Weight vector (learned scale parameters)
 * @param size Length of the vectors
 */
void rmsnorm(float* out, float* x, float* w, int size);

/**
 * @brief Convert logits (e.g. attention scores) into a probability distribution.
 * @param x Input vector (overwritten in-place with softmaxed values)
 * @param size Number of elements in vector
 */
void softmax(float* x, int size);

/**
 * @brief Multiply quantized matrix W[d × n] by quantized input x[n], output float[d].
 *
 * Computes: out[i] = sum_j ( x.q[j] * w.q[i*n + j] ) * x.s[j/GS] * w.s[(i*n + j)/GS]
 *
 * This function assumes both x and w are quantized with group size GS.
 * The inner product is accumulated in int32 then dequantized.
 *
 * @param out Output vector of size d (float32)
 * @param x   Quantized input vector of size n
 * @param w   Quantized weight matrix of shape [d][n] (flattened row-major)
 * @param n   Input dimension
 * @param d   Output dimension
 */
void matmul(float* out, Q8Tensor* x, Q8Tensor* w, int n, int d);

/**
 * @brief Applies rotary positional embeddings (RoPE) in-place to a vector x.
 *
 * Assumes x is split into real and imaginary halves:
 *   - x[0 .. head_dim/2 - 1] → real part
 *   - x[head_dim/2 .. head_dim - 1] → imaginary part
 *
 * Each complex pair (real, imag) is rotated by a fixed angle depending on position:
 *
 *   angle_i = pos * (1000000 ^ -(i / (head_dim/2)))
 *
 * This rotates each vector slice by a different frequency — used for relative attention.
 *
 * @param x         Attention vector (query or key), modified in-place
 * @param head_dim  Dimension of the attention head (must be even)
 * @param pos       Token position in the sequence
 */
void rotary(float* x, int head_dim, int pos);

float sigmoid(float x);
float silu(float x);

/**
 * @brief Applies the SwiGLU activation function element-wise.
 *
 * Computes:
 *     x1[i] = silu(x1[i]) * x3[i]
 *
 * where silu(x) = x / (1 + exp(-x)) is the Sigmoid Linear Unit,
 * and ⊙ represents element-wise multiplication.
 *
 * This is typically used in transformer FFNs where:
 *     - x1 holds the result of W₁x
 *     - x3 holds the result of W₃x
 *
 * The result is stored in-place in x1.
 *
 * @param x1   Pointer to first input array (will be overwritten).
 * @param x3   Pointer to second input array (gating vector).
 * @param size Number of elements in both arrays.
 */
void swiglu(float* x1, float* x3, int size);

/**
 * @brief Performs multi-head self-attention for a single transformer layer at a given time step.
 *
 * For each head:
 *   - Computes dot products between current query vector q and past keys k_i
 *   - Applies softmax over dot products to get attention weights
 *   - Computes a weighted sum over the corresponding values v_i
 *   - Stores the result in s->r (attention output buffer)
 *
 * Assumes:
 *   - Queries for this position already populated in s->q
 *   - Keys/values populated in s->k_cache / s->v_cache
 *   - Rotary embeddings already applied to q before this call
 *
 * @param t    Pointer to transformer instance.
 * @param l    Layer index
 * @param pos  Current token position (for causal masking)
 */
void attention(Transformer* t, int l, int pos);

#endif // QWEN_FORWARD_H
