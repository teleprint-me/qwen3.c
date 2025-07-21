/**
 * @file include/q8.h
 */

#ifndef QWEN_Q8_H
#define QWEN_Q8_H

#include <stdint.h>

#define Q8_MAX 127.0f

/**
 * @struct Q8Tensor
 * @brief A quantized tensor with int8 values and per-block scale factors.
 */
typedef struct Q8Tensor {
    float* s; ///< scaling factors per block
    int8_t* q; ///< quantized values
} Q8Tensor;

/**
 * Quantizes `n` float32 values into Q8 format using block-wise scaling.
 * `block_size` must evenly divide `n`.
 */
void q8_quantize(Q8Tensor* qt, float* x, int n, int block_size);

/**
 * Dequantizes `n` Q8 values back into float32.
 */
void q8_dequantize(Q8Tensor* qt, float* x, int n, int block_size);

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
 * @param size Number of int8_t elements in each tensor (must be divisible by GS)
 * @return Q8Tensor array with n entries (caller must free)
 */
Q8Tensor* q8_tensor_mmap(void** buffer, int n, int n_int8, int block_size);

#endif // QWEN_Q8_H
