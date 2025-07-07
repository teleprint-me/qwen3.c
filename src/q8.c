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

Q8Tensor* q8_tensor(void** b, int n, int size) {
    void* c = *b; // current pos
    Q8Tensor* qt = calloc(n, sizeof(Q8Tensor));
    if (!qt) {
        return NULL;
    }

    // Buffer must be read linearly
    for (int i = 0; i < n; i++) {
        // map q8 values
        qt[i].q = (int8_t*) c;
        c = (int8_t*) c + size;

        // map scalars
        qt[i].s = (float*) c;
        c = (float*) c + size / GS;
    }

    *b = c; // advance buf
    return qt;
}
