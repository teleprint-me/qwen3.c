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
 * Initializes a Q8Tensor view from raw memory.
 *   - `W` should point to memory of size (n * stride + n * sizeof(float))
 *   - `n` is the number of groups
 *   - `stride` is the number of int8 values per group
 *   - Returns a Q8Tensor pointing into `*w` (memory is not copied)
 */
Q8Tensor* q8_tensor(void** W, int n, int stride);

#ifdef __cplusplus
}
#endif

#endif // QWEN_Q8_H
