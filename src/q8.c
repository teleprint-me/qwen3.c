/// @file src/q8.c
#include "q8.h"

extern int GS = 1;

void q8_quantize(Q8Tensor *qx, float *x, int n) {
    const int num_groups = n / GS;
    const float Q_MAX = 127.0f;

    for (int group = 0; group < num_groups; group++) {
        float* xg  = x + group * GS;
        int8_t* qg = qx->q + group * GS;

        // Find max absolute value
        float wmax = fabsf(xg[0]);
        #pragma omp simd reduction(max:wmax)
        for (int i = 1; i < GS; i++) {
            wmax = fmaxf(wmax, fabsf(xg[i]));
        }

        float scale = (wmax == 0.0f) ? 1e-6f : (wmax / Q_MAX); // avoid div by 0
        qx->s[group] = scale;

        /// @note Clamp to [-127, 127] to avoid int8 overflow on rare large values.
        #pragma omp parallel for
        for (int i = 0; i < GS; i++) {
            float q = xg[i] / scale;
            qg[i] = (int8_t) fminf(fmaxf(roundf(q), -Q_MAX), Q_MAX);
        }
    }
}

void q8_dequantize(Q8Tensor *qx, float *x, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        x[i] = qx->q[i] * qx->s[i / GS];
    }
}

Q8Tensor *q8_tensor(void **ptr, int n, int size_each) {
    void *p = *ptr;
    Q8Tensor *res = malloc(n * sizeof(Q8Tensor));

    for (int i = 0; i < n; i++) {
        /* map quantized int8 values*/
        res[i].q = (int8_t*)p;
        p = (int8_t*)p + size_each;
        /* map scale factors */
        res[i].s = (float*)p;
        p = (float*)p + size_each / GS;
    }
    *ptr = p; // advance ptr to current position
    return res;
}
