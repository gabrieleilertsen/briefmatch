/**
 * \brief Cuda kernels for the BriefMatch class.
 *
 * Cuda GPU kernels used by BriefMatch in order to compute binary features and
 * perform per-pixel matching.
 *
 *
 * This file is part of the BriefMatch package.
 * -----------------------------------------------------------------------------
 * Copyright (c) 2017, The BriefMatch authors.
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without 
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice, 
 *    this list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software 
 *    without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
 * POSSIBILITY OF SUCH DAMAGE.
 * -----------------------------------------------------------------------------
 *
 * \author Gabriel Eilertsen, gabriel.eilertsen@liu.se
 *
 * \date Jul 2017
 */

#ifndef BRIEF_MATCH_CUDA_H
#define BRIEF_MATCH_CUDA_H

#include "util_cuda.h"

#ifndef FLT_MAX
#define FLT_MAX 1E+30;
#endif

#ifndef sint
typedef short int sint;
#endif

#ifndef uchar
typedef unsigned char uchar;
#endif

#define PATCH_MAX 2048

extern "C"
{
    void CUDA_initRandom(curandState *state, int sx, int sy);
    void CUDA_binaryFeatures(const float* I, unsigned int* F,
                             int sx, int sy, 
                             int N, int N_space);
    void CUDA_initialize(const float* I1, const float* I2,
                         unsigned int* F1, unsigned int* F2,
                         float *D, sint *NX, sint *NY,
                         int sx, int sy,
                         bool first, int N, int N_space);
    void CUDA_propagate(const float* I1, const float* I2,
                        unsigned int* F1, unsigned int* F2,
                        float *D, sint *NX, sint *NY,
                        int sx, int sy,
                        int pd, int jfMax, bool doubleProp,
                        int N, int N_space);
    void CUDA_randomSearch(const float* I1, const float* I2,
                           unsigned int* F1, unsigned int* F2,
                           float *D, sint *NX, sint *NY, curandState *state,
                           int sx, int sy, float dist,
                           int N, int N_space);
    void CUDA_cbFilter(const float *Ui, const float *Vi,
                       float *Uo, float *Vo,
                       const float *Uim, const float *Vim,
                       const float *I,
                       const float sigmaEPE, const float sigmaI, const float sigmaS,
                       const int sx, const int sy);
    void CUDA_copyPatchToConst(float *patch);
}

#endif //BRIEF_MATCH_CUDA_H