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
// Complex floating-point dot product (AVX + FMA3)
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
// (a + jb)(c + jd) = (ac - bd) + j(ad + bc)
//
// Despite the historical association with AVX2 (both arrived on Haswell in
// the same generation), this kernel only uses 256-bit float ops (load,
// store, shuffle, addsub) and _mm256_fmadd_ps, none of which need AVX2
// itself (AVX2 is mainly about 256-bit integer ops); it needs only AVX
// registers plus the FMA3 encoding, which is why it is gated on
// BUILD_FMA3/LIQUID_RUNTIME_FMA3 rather than AVX2. The real and imaginary
// coefficient products are accumulated separately (sumi = sum of v*hi,
// sumq = sum of v*hq) using fused multiply-adds, then combined once at the
// end with a single shuffle + addsub. Four independent accumulator pairs
// keep the loop bound by FMA throughput rather than by the latency of one
// dependent accumulator chain.
// Below the plain-AVX dispatch threshold (_q->n < 64) that setup and fold
// cost more than they save, so this defers to the single-accumulator
// dotprod_cccf_execute_avx_1() instead of running its own small-n path.
int __attribute__((target("avx,fma")))
dotprod_cccf_execute_fma3(dotprod_cccf    _q,
                          float complex * _x,
                          float complex * _y)
{
    liquid_log_trace("dotprod_cccf_execute_fma3()");

    if (_q->n < 64)
        return dotprod_cccf_execute_avx_1(_q, _x, _y);

    // type cast input as floating point array; double effective length
    float * x = (float*) _x;
    unsigned int n = 2*_q->n;

    // four independent accumulator pairs (in-phase / quadrature)
    __m256 si0 = _mm256_setzero_ps(), sq0 = _mm256_setzero_ps();
    __m256 si1 = _mm256_setzero_ps(), sq1 = _mm256_setzero_ps();
    __m256 si2 = _mm256_setzero_ps(), sq2 = _mm256_setzero_ps();
    __m256 si3 = _mm256_setzero_ps(), sq3 = _mm256_setzero_ps();

    // process 32 floats per iteration (4 vectors of 8 => 16 complex samples)
    unsigned int r = (n >> 5) << 5;
    unsigned int i;
    for (i=0; i<r; i+=32) {
        __m256 v0 = _mm256_loadu_ps(&x[i   ]);
        __m256 v1 = _mm256_loadu_ps(&x[i+ 8]);
        __m256 v2 = _mm256_loadu_ps(&x[i+16]);
        __m256 v3 = _mm256_loadu_ps(&x[i+24]);

        si0 = _mm256_fmadd_ps(v0, _mm256_load_ps(&_q->hi[i   ]), si0);
        si1 = _mm256_fmadd_ps(v1, _mm256_load_ps(&_q->hi[i+ 8]), si1);
        si2 = _mm256_fmadd_ps(v2, _mm256_load_ps(&_q->hi[i+16]), si2);
        si3 = _mm256_fmadd_ps(v3, _mm256_load_ps(&_q->hi[i+24]), si3);

        sq0 = _mm256_fmadd_ps(v0, _mm256_load_ps(&_q->hq[i   ]), sq0);
        sq1 = _mm256_fmadd_ps(v1, _mm256_load_ps(&_q->hq[i+ 8]), sq1);
        sq2 = _mm256_fmadd_ps(v2, _mm256_load_ps(&_q->hq[i+16]), sq2);
        sq3 = _mm256_fmadd_ps(v3, _mm256_load_ps(&_q->hq[i+24]), sq3);
    }

    // fold the four accumulator pairs into one
    si0 = _mm256_add_ps(_mm256_add_ps(si0, si1), _mm256_add_ps(si2, si3));
    sq0 = _mm256_add_ps(_mm256_add_ps(sq0, sq1), _mm256_add_ps(sq2, sq3));

    // process remaining blocks of 8
    unsigned int t = (n >> 3) << 3;
    for (; i<t; i+=8) {
        __m256 v = _mm256_loadu_ps(&x[i]);
        si0 = _mm256_fmadd_ps(v, _mm256_load_ps(&_q->hi[i]), si0);
        sq0 = _mm256_fmadd_ps(v, _mm256_load_ps(&_q->hq[i]), sq0);
    }

    // swap real/imag within each complex pair, then combine:
    //  even lanes -> si - sq  (real),  odd lanes -> si + sq  (imag)
    sq0 = _mm256_shuffle_ps(sq0, sq0, _MM_SHUFFLE(2,3,0,1));
    __m256 res = _mm256_addsub_ps(si0, sq0);

    // aligned output array
    float w[8] __attribute__((aligned(32)));
    _mm256_store_ps(w, res);

    float complex total = (w[0] + w[2] + w[4] + w[6]) +
                          (w[1] + w[3] + w[5] + w[7]) * _Complex_I;

    // cleanup (i is in float units; i/2 is the complex tap index)
    unsigned int k;
    for (k=i/2; k<_q->n; k++)
        total += _x[k] * ( _q->hi[2*k] + _q->hq[2*k]*_Complex_I );

    // set return value
    *_y = total;
    return LIQUID_OK;
}

// build guard
#else

// invalidated
int dotprod_cccf_execute_fma3(dotprod_cccf    _q,
                              float complex * _x,
                              float complex * _y)
{
    return liquid_error(LIQUID_EICONFIG,"fma3 extensions not available");
}

// build guard
#endif
