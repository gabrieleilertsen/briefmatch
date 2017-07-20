/**
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

#ifndef BRIEF_MATCH_CUDA_CU
#define BRIEF_MATCH_CUDA_CU

#include "briefmatch_cuda.h"

__constant__ float const_sampPatch[PATCH_MAX];

__global__ void initRandom(curandState *state, unsigned int sx, unsigned int sy)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= sx || y >= sy)
        return;

    int id = x + y * sx;
    
    // per pixel seed
    curand_init(id, 0, 0, &state[id]);
}

__device__ float featDist(unsigned int* F1, unsigned int* F2, float dmax,
                          int offset, int nx, int ny, int sx, int N, int N_space)
{
    float ddmax = N*dmax;
    sint dist = 0;
    
    for (unsigned int s=0; s<N_space; s++)
    {
        //xor operation of bit features
        unsigned int bd = (F1[offset+s] ^ F2[offset+s + nx*N_space + ny*sx*N_space]);
        
        //bitsum of xor (Hamming distance)
        dist += __popc(bd);
        
        //early stopping
        if (dist > ddmax)
            return -1;
    }
    
    return float(dist)/N;
}

__global__ void binaryFeatures(const float* I, unsigned int* F,
                               int sx, int sy, int N, int N_space)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= sx || y >= sy)
        return;

    int offset = x + y * sx;
    int offsetF = x * N_space + y * N_space * sx;
    
    float i1, i2;
    unsigned int ff;
    
    int c = 1, n = -1, ox1, ox2, oy1, oy2;
    for (unsigned int s=0; s<N; s++)
    {
        if (s%32 == 0)
        {
            if (n>=0)
                F[offsetF + n] = ff;
            
            ff = 0; c = 0;
            n++;
        }
        
        // sampling of patch
        ox1 = max(-x, min( sx-x-1, int(const_sampPatch[4*s])));
        oy1 = max(-y, min( sy-y-1, int(const_sampPatch[4*s+1])));
        ox2 = max(-x, min( sx-x-1, int(const_sampPatch[4*s+2])));
        oy2 = max(-y, min( sy-y-1, int(const_sampPatch[4*s+3])));
        
        i1 = I[offset + ox1 + oy1*sx];
        i2 = I[offset + ox2 + oy2*sx];
        
        // difference test for feature bit vector
        ff += ((i1 > i2) << c);
        
        c++;
    }
    F[offsetF + n] = ff;
}

__device__ bool insert(const float* I1, const float* I2, unsigned int* F1, unsigned int* F2,
                       float *D, sint *NX, sint *NY,
                       int offset, int offsetF, int nx, int ny, int sx, int sy,
                       int N, int N_space)
{
    if (nx == NX[offset] && ny == NY[offset])
        return false;
    
    float dst = featDist(F1,F2,D[offset],offsetF,nx,ny,sx,N,N_space);
        
    if (dst < 0)
        return false;
    else if (dst < D[offset])
    {
        
        D[offset]  = dst;
        NX[offset] = nx;
        NY[offset] = ny;
    }
    
    return true;
}

__global__ void init(const float* I1, const float* I2,
                     unsigned int* F1, unsigned int* F2,
                     float *D, sint *NX, sint *NY,
                     int sx, int sy,
                     bool first, int N, int N_space)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= sx || y >= sy)
        return;

    int offset = x + y * sx;
    int offsetF = x * N_space + y * N_space * sx;
    
    if (first)
    {
        D[offset] = featDist(F1,F2,FLT_MAX,offsetF,0,0,sx,N,N_space);
        NX[offset] = 0;
        NY[offset] = 0;
    }
    else
    {
        sint nx0, ny0, nx, ny;
        
        nx0 = NX[offset];
        ny0 = NY[offset];
        nx = min(max(NX[offset+nx0+ny0*sx], -x), sx-x-1),
        ny = min(max(NY[offset+nx0+ny0*sx], -y), sy-y-1);
        
        D[offset] = featDist(F1,F2,FLT_MAX,offsetF,nx,ny,sx,N,N_space);
        
        NX[offset] = nx;
        NY[offset] = ny;
    }
}

__global__ void propagate(const float* I1, const float* I2,
                          unsigned int* F1, unsigned int* F2,
                          float *D, sint *NX, sint *NY,
                          int sx, int sy,
                          int pd, int jfMax, bool doubleProp,
                          int N, int N_space)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (x >= sx || y >= sy)
        return;

    int offset = x + y * sx;
    int offsetF = x * N_space + y * N_space * sx;
    
    sint nx, ny;
    int doff, dstart = doubleProp ? -1 : 1;
    
    // jump flooding scheme
    for (int p=jfMax; p>0; p/=2)
    {
        for (int d=dstart; d<=1; d+=2)
        {
            for (int xy=0; xy<=1; xy++)
            {
                doff = pd*p*d*xy + sx*pd*p*d*(1-xy);
                nx = min(max(NX[offset+doff], -x), sx-x-1);
                ny = min(max(NY[offset+doff], -y), sy-y-1);
                
                insert(I1, I2, F1, F2, D, NX, NY, offset, offsetF, nx, ny, sx, sy, N, N_space);
            }
        }
    }
}

__global__ void randomSearch(const float* I1, const float* I2,
                             unsigned int* F1, unsigned int* F2,
                             float *D, sint *NX, sint *NY, curandState *state,
                             int sx, int sy, float dist,
                             int N, int N_space)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (x >= sx || y >= sy)
        return;

    int offset = x + y * sx;
    int offsetF = x * N_space + y * N_space * sx;
    
    curandState localState = state[offset];
    
    // random neighbor
    sint nx = min(max((int)round(dist*curand_normal(&localState)), -x), sx-x-1);
    sint ny = min(max((int)round(dist*curand_normal(&localState)), -y), sy-y-1);

    insert(I1, I2, F1, F2, D, NX, NY, offset, offsetF, nx, ny, sx, sy, N, N_space);
    
    state[offset] = localState;
}

static __global__ void cbFilter(const float *Ui, const float *Vi,        // input flow vectors
                                float *Uo, float *Vo,                    // output flow vectors
                                const float *Uim, const float *Vim,      // median filtered flow vectors
                                const float *I,                          // input image
                                const float sigmaEPE, const float sigmaI, const float sigmaS, // gaussian kernel sizes
                                const int sx, const int sy)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (x >= sx || y >= sy)
        return;

    int offset = x + y*sx;
    
    if (x >= sx || y >= sy)
        return;
    
    int s = ceil(2.5*sqrt(sigmaS)), offset1;
    float um0 = Uim[offset], u1, du,
          vm0 = Vim[offset], v1, dv,
          I0 = I[offset]/255, I1;
          
    float w, dEPE, dI;

    float uf = 0, vf = 0, W = 0;
    for (int i=-s; i<s; i++)
        for (int j=-s; j<s; j++)
        {
            offset1 = min(sx-1,max(0,x+i)) + min(sy-1,max(0,y+j))*sx;
            u1 = Ui[offset1];
            v1 = Vi[offset1];
            I1 = I[offset1]/255;
            
            du = u1-um0; dv = v1-vm0;
            
            dEPE = (du*du+dv*dv);
            dI = (I1-I0);

            // the cross-trilateral kernel
            w = exp(-(dEPE)/(2*sigmaEPE)) * exp(-(dI*dI)/(2*sigmaI)) * exp(-(i*i + j*j)/(2*sigmaS));
            
            uf += w*u1;
            vf += w*v1;
            W += w;
        }
    
    Uo[offset] = uf/max(1e-5,W);
    Vo[offset] = vf/max(1e-5,W);
}


extern "C"
{
void CUDA_initRandom(curandState *state, int sx, int sy)
{
    dim3 blocks((sx+BLOCK_X-1)/BLOCK_X,(sy+BLOCK_Y-1)/BLOCK_Y);
    dim3 threads(BLOCK_X,BLOCK_Y);
    initRandom<<<blocks, threads>>>(state, sx, sy);
    HANDLE_ERROR( cudaPeekAtLastError() );
    HANDLE_ERROR( cudaDeviceSynchronize() );
}

void CUDA_binaryFeatures(const float* I, unsigned int* F,
                         int sx, int sy, int N, int N_space)
{
    dim3 blocks((sx+BLOCK_X-1)/BLOCK_X,(sy+BLOCK_Y-1)/BLOCK_Y);
    dim3 threads(BLOCK_X,BLOCK_Y);
    binaryFeatures<<<blocks,threads>>>(I, F, sx, sy, N, N_space);
    HANDLE_ERROR( cudaPeekAtLastError() );
    HANDLE_ERROR( cudaDeviceSynchronize() );
}

void CUDA_initialize(const float* I1, const float* I2,
                     unsigned int* F1, unsigned int* F2,
                     float *D, sint *NX, sint *NY,
                     int sx, int sy,
                     bool first, int N, int N_space)
{
    dim3 blocks((sx+BLOCK_X-1)/BLOCK_X,(sy+BLOCK_Y-1)/BLOCK_Y);
    dim3 threads(BLOCK_X,BLOCK_Y);
    init<<<blocks,threads>>>(I1, I2, F1, F2, D, NX, NY, sx, sy, first, N, N_space);
    HANDLE_ERROR( cudaPeekAtLastError() );
    HANDLE_ERROR( cudaDeviceSynchronize() );
}

void CUDA_propagate(const float* I1, const float* I2,
                    unsigned int* F1, unsigned int* F2,
                    float *D, sint *NX, sint *NY,
                    int sx, int sy,
                    int pd, int jfMax, bool doubleProp,
                    int N, int N_space)
{
    dim3 blocks((sx+BLOCK_X-1)/BLOCK_X,(sy+BLOCK_Y-1)/BLOCK_Y);
    dim3 threads(BLOCK_X,BLOCK_Y);
    
    propagate<<<blocks,threads>>> (I1, I2, F1, F2, D, NX, NY, sx, sy, pd, jfMax, doubleProp, N, N_space);
    HANDLE_ERROR( cudaPeekAtLastError() );
    HANDLE_ERROR( cudaDeviceSynchronize() );
}

void CUDA_randomSearch(const float* I1, const float* I2,
                       unsigned int* F1, unsigned int* F2,
                       float *D, sint *NX, sint *NY, curandState *state,
                       int sx, int sy, float dist,
                       int N, int N_space)
{
    dim3 blocks((sx+BLOCK_X-1)/BLOCK_X,(sy+BLOCK_Y-1)/BLOCK_Y);
    dim3 threads(BLOCK_X,BLOCK_Y);
    randomSearch<<<blocks,threads>>>(I1, I2, F1, F2, D, NX, NY, state, sx, sy, dist, N, N_space);
    HANDLE_ERROR( cudaPeekAtLastError() );
    HANDLE_ERROR( cudaDeviceSynchronize() );
}

void CUDA_cbFilter(const float *Ui, const float *Vi,
                   float *Uo, float *Vo,            
                   const float *Uim, const float *Vim,
                   const float *I,
                   const float sigmaEPE, const float sigmaI, const float sigmaS,
                   const int sx, const int sy)
{
    dim3 blocks((sx+BLOCK_X-1)/BLOCK_X,(sy+BLOCK_Y-1)/BLOCK_Y);
    dim3 threads(BLOCK_X,BLOCK_Y);
    cbFilter<<<blocks,threads>>>(Ui, Vi, Uo, Vo, Uim, Vim, I, sigmaEPE, sigmaI, sigmaS, sx, sy);
    HANDLE_ERROR( cudaPeekAtLastError() );
    HANDLE_ERROR( cudaDeviceSynchronize() );
}

void CUDA_copyPatchToConst(float *patch)
{
    HANDLE_ERROR( cudaMemcpyToSymbol( const_sampPatch, patch, sizeof(float)*PATCH_MAX) );
}
}

#endif //BRIEF_MATCH_CUDA_CU



