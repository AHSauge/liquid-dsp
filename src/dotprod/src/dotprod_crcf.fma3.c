/*
 * Copyright (c) 2007 - 2026 Joseph Gaeddert
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

//
// Floating-point dot product, real coefficients, complex input (AVX + FMA3)
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "liquid.internal.h"

// build guard
#if BUILD_FMA3

// include proper SIMD extensions for x86 AVX/FMA3
#include <immintrin.h>

// use AVX/FMA3 extensions
//
// Despite the historical association with AVX2 (both arrived on Haswell in
// the same generation), this kernel only uses 256-bit float loads/stores
// and _mm256_fmadd_ps, none of which need AVX2 itself (AVX2 is mainly about
// 256-bit integer ops); it needs only AVX registers plus the FMA3 encoding,
// which is why it is gated on BUILD_FMA3/LIQUID_RUNTIME_FMA3 rather than
// AVX2. The complex input is treated as a real array of twice the length;
// the real coefficients are stored duplicated (h[2i] == h[2i+1]) so a
// single real multiply mixes each coefficient with both the in-phase and
// the quadrature parts of the sample. As in the rrrf kernel this uses a
// fused multiply-add and four independent accumulators to stay
// throughput-bound.
// Below the plain-AVX dispatch threshold (_q->n < 64) that setup and fold
// cost more than they save, so this defers to the single-accumulator
// dotprod_crcf_execute_avx_1() instead of running its own small-n path.
int __attribute__((target("avx,fma")))
dotprod_crcf_execute_fma3(dotprod_crcf    _q,
                          float complex * _x,
                          float complex * _y)
{
    liquid_log_trace("dotprod_crcf_execute_fma3()");

    if (_q->n < 64)
        return dotprod_crcf_execute_avx_1(_q, _x, _y);

    // type cast input as floating point array; double effective length
    float * x = (float*) _x;
    unsigned int n = 2*_q->n;

    // four independent accumulators to hide FMA latency
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();

    // process 32 floats per iteration (4 vectors of 8)
    unsigned int r = (n >> 5) << 5;
    unsigned int i;
    for (i=0; i<r; i+=32) {
        sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(&x[i   ]), _mm256_load_ps(&_q->h[i   ]), sum0);
        sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(&x[i+ 8]), _mm256_load_ps(&_q->h[i+ 8]), sum1);
        sum2 = _mm256_fmadd_ps(_mm256_loadu_ps(&x[i+16]), _mm256_load_ps(&_q->h[i+16]), sum2);
        sum3 = _mm256_fmadd_ps(_mm256_loadu_ps(&x[i+24]), _mm256_load_ps(&_q->h[i+24]), sum3);
    }

    // process remaining blocks of 8
    unsigned int t = (n >> 3) << 3;
    for (; i<t; i+=8)
        sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(&x[i]), _mm256_load_ps(&_q->h[i]), sum0);

    // fold the four accumulators into one
    sum0 = _mm256_add_ps(sum0, sum1);
    sum2 = _mm256_add_ps(sum2, sum3);
    sum0 = _mm256_add_ps(sum0, sum2);

    // aligned output array
    float w[8] __attribute__((aligned(32)));
    _mm256_store_ps(w, sum0);

    // even lanes -> in-phase, odd lanes -> quadrature
    float wi = w[0] + w[2] + w[4] + w[6];
    float wq = w[1] + w[3] + w[5] + w[7];

    // cleanup (note: n _must_ be even)
    for (; i<n; i+=2) {
        wi += x[i  ] * _q->h[i  ];
        wq += x[i+1] * _q->h[i+1];
    }

    // set return value
    *_y = wi + _Complex_I*wq;
    return LIQUID_OK;
}

// build guard
#else

// invalidated
int dotprod_crcf_execute_fma3(dotprod_crcf    _q,
                              float complex * _x,
                              float complex * _y)
{
    return liquid_error(LIQUID_EICONFIG,"fma3 extensions not available");
}

// build guard
#endif
