/**
 * @file include/forward.h
 * @brief Transformer forward pass and core operations (Qwen-compatible).
 */

#ifndef QWEN_FORWARD_H
#define QWEN_FORWARD_H

#include "q8.h"
#include "checkpoint.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Core Transformer Operations
 */

/**
 * @brief Performs a root-mean-square normalization followed by learned scaling.
 *
 * Each element is normalized as:
 *     out[i] = w[i] * x[i] / sqrt(mean(x^2) + ε)
 *
 * @param out  Output buffer (same size as x)
 * @param x    Input vector to normalize
 * @param w    Weight vector of learned scaling parameters
 * @param size Number of elements in x and w
 */
void rmsnorm(float* out, float* x, float* w, int size);

/**
 * @brief Applies softmax in-place to a vector.
 *
 * Converts raw scores into a probability distribution using:
 *     softmax(x_i) = exp(x_i - max(x)) / sum_j(exp(x_j - max(x)))
 *
 * @param x    Input vector (overwritten with normalized output)
 * @param size Number of elements in x
 */
void softmax(float* x, int size);

/**
 * @brief Performs matrix-vector multiplication with quantized inputs and weights.
 *
 * Computes:
 *     out[i] = ∑_j (x.q[j] * w.q[i*n + j]) * x.s[j/GS] * w.s[(i*n + j)/GS]
 *
 * Both operands are assumed to use the same group size `GS` for quantization.
 *
 * @param out Output vector (size d, float32)
 * @param x   Quantized input vector of size n
 * @param w   Quantized weight matrix [d × n] (row-major layout)
 * @param n   Input dimension
 * @param d   Output dimension
 */
void matmul(float* out, Q8Tensor* x, Q8Tensor* w, int n, int d);

/**
 * @brief Applies rotary positional embeddings in-place.
 *
 * Each input vector is split into two halves and rotated in 2D space:
 *     angle_i = pos * 1000000^(-i / (head_dim / 2))
 *
 * @param x        Input buffer (query or key vector), modified in-place
 * @param head_dim Head dimension (must be even)
 * @param pos      Token position in the sequence
 */
void rotary(float* x, int head_dim, int pos);

/**
 * Activation Functions
 */

/**
 * @brief Sigmoid function: σ(x) = 1 / (1 + exp(-x))
 *
 * @param x Input scalar
 * @return  Sigmoid activation
 */
float sigmoid(float x);

/**
 * @brief SiLU activation: silu(x) = x * sigmoid(x)
 *
 * @param x Input scalar
 * @return  SiLU activation
 */
float silu(float x);

/**
 * @brief SwiGLU non-linearity: x1 = silu(x1) ⊙ x3
 *
 * Element-wise:
 *     x1[i] = silu(x1[i]) * x3[i]
 *
 * Commonly used in transformer FFNs (Gated Linear Units).
 *
 * @param x1   First input vector (will be overwritten with result)
 * @param x3   Second input vector (gating vector)
 * @param size Number of elements
 */
void swiglu(float* x1, float* x3, int size);

/**
 * Attention & Forward
 */

/**
 * @brief Multi-head self-attention for a single transformer layer.
 *
 * - Assumes rotary embeddings and RMSNorm already applied to q/k.
 * - Updates attention buffer and writes attention output to s->r.
 * - Accumulates across all heads using causal masking.
 *
 * @param t    Pointer to Transformer instance
 * @param l    Layer index
 * @param pos  Current sequence position
 */
void attention(Transformer* t, int l, int pos);

/**
 * @brief Runs a full forward pass through the transformer.
 *
 * Computes:
 *     logits = model(x₀, ..., x_pos)
 *
 * Final output is the unnormalized logits at position `pos`,
 * written to `state->logits` and returned.
 *
 * @param t     Pointer to Transformer instance
 * @param token Input token ID at current timestep
 * @param pos   Current position in the sequence
 * @return      Pointer to logits buffer (float[p->vocab_size])
 */
float* forward(Transformer* t, int token, int pos);

#ifdef __cplusplus
}
#endif

#endif // QWEN_FORWARD_H
