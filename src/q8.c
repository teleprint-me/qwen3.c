/// @file src/q8.c
#include "q8.h"
#include <math.h>

void q8_quantize(Q8Tensor* qt, float* x, int n, int block_size) {
    const int num_groups = n / block_size;

    for (int group = 0; group < num_groups; group++) {
        float* xg = x + group * block_size;
        int8_t* qg = qt->q + group * block_size;

        // Find max absolute value
        float wmax = fabsf(xg[0]);
#pragma omp simd reduction(max : wmax)
        for (int i = 1; i < block_size; i++) {
            wmax = fmaxf(wmax, fabsf(xg[i]));
        }

        float scale = (wmax == 0.0f) ? 1e-6f
                                     : (wmax / Q8_MAX);  // avoid div by 0
        qt->s[group] = scale;

/// @note Clamp to [-127, 127] to avoid int8 overflow on rare large values.
#pragma omp parallel for
        for (int i = 0; i < block_size; i++) {
            float q = xg[i] / scale;
            qg[i] = (int8_t) fminf(fmaxf(roundf(q), -Q8_MAX), Q8_MAX);
        }
    }
}

void q8_dequantize(Q8Tensor* qt, float* x, int n, int block_size) {
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        x[i] = qt->q[i] * qt->s[i / block_size];
    }
}
