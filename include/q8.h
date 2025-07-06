/// @file include/q8.h
#ifndef QWEN_Q8_H
#define QWEN_Q8_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

extern int GS; // global quantization group size

typedef struct {
    int8_t* q; // quantized values
    float* s; // scaling factors per group
} Q8Tensor;

/**
 * Quantizes `n` float32 values into Q8 format using group-wise scaling.
 * Assumes `GS` evenly divides `n`.
 */
void q8_quantize(Q8Tensor* qt, float* x, int n);

/**
 * Dequantizes `n` Q8 values back into float32.
 */
void q8_dequantize(Q8Tensor* qt, float* x, int n);

/**
 * Parses an array of Q8_0 quantized tensors from flat memory.
 *
 * Each tensor is stored as:
 *   - [int8_t[size]]: flattened weight
 *   - [float[size / GS]]: scale factors (1 per group)
 *
 * The input `*X` is expected to point to the start of this data.
 * On return, `*X` will be advanced past the parsed region.
 *
 * @param b Pointer to raw memory buffer (usually mapped model)
 * @param n Number of tensors to parse
 * @param size Number of int8_t elements in each tensor (must be divisible by GS)
 * @return Q8Tensor array with n entries (caller must free)
 */
Q8Tensor* q8_tensor(void** b, int n, int size);

#ifdef __cplusplus
}
#endif

#endif // QWEN_Q8_H
