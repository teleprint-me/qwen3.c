/// @file include/q8.h
#ifndef QWEN_Q8_H
#define QWEN_Q8_H

#include <stdint.h>

extern int GS; // group size for quantization (default: 1)

typedef struct Q8Tensor {
  int8_t *q; // quantized values
  float *s;  // scaling factors
} Q8Tensor;

void q8_quantize(Q8Tensor *qt, float *x, int n);
void q8_dequantize(Q8Tensor *qt, float *x, int n);

/**
 * Initializes a Q8Tensor from raw memory:
 *   - `w` should point to a buffer (e.g., from mmap or malloc)
 *   - `n` is the number of groups
 *   - `group_size` is the number of int8 values per group (usually equal to group size)
 *
 * Returns a pointer to a Q8Tensor where:
 *   - q points to int8 values
 *   - s points to group scaling factors
 */
Q8Tensor *q8_tensor(void **w, int n, int group_size);

#endif // QWEN_Q8_H
