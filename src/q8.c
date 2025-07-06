/// @file src/q8.c
#include "q8.h"
#include <math.h>
#include <stdlib.h>

extern int GS = 1;

void q8_quantize(Q8Tensor* qt, float* x, int n) {
    const int num_groups = n / GS;
    const float Q_MAX = 127.0f;

    for (int g = 0; g < num_groups; g++) {
        float* xg = x + g * GS;
        int8_t* qg = qt->q + g * GS;

        // Find max absolute value
        float wmax = fabsf(xg[0]);
#pragma omp simd reduction(max : wmax)
        for (int i = 1; i < GS; i++) {
            wmax = fmaxf(wmax, fabsf(xg[i]));
        }

        float scale = (wmax == 0.0f) ? 1e-6f : (wmax / Q_MAX); // avoid div by 0
        qt->s[g] = scale;

/// @note Clamp to [-127, 127] to avoid int8 overflow on rare large values.
#pragma omp parallel for
        for (int i = 0; i < GS; i++) {
            float q = xg[i] / scale;
            qg[i] = (int8_t) fminf(fmaxf(roundf(q), -Q_MAX), Q_MAX);
        }
    }
}

void q8_dequantize(Q8Tensor* qt, float* x, int n) {
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        x[i] = qt->q[i] * qt->s[i / GS];
    }
}

Q8Tensor* q8_tensor(void** W, int n, int stride) {
    void* p = *W;
    Q8Tensor* res = malloc(n * sizeof(Q8Tensor));

    for (int i = 0; i < n; i++) {
        // map quantized int8 values
        res[i].q = (int8_t*) p;
        p = (int8_t*) p + stride;

        // map scale factors
        res[i].s = (float*) p;
        p = (float*) p + stride / GS;
    }
    *W = p; // advance ptr to current position
    return res;
}
