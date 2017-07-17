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

#ifndef UTIL_CUDA_CU
#define UTIL_CUDA_CU

#include "util_cuda.h"
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

__constant__ float const_kernel[CONST_KERNEL_SIZE];

__global__ void filterConst(const float *Li, float *Lo,
                            int ks, int dir, int sx, int sy)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * sx;

    if (x >= sx || y >= sy)
        return;

    float s = 0;
    int ks2 = ks/2;
    
    if (dir)
    {
        for (int i=0; i<ks; i++)
            s += const_kernel[i]*Li[min(sx-1,max(0,x+i-ks2)) + y*sx];
    }
    else
    {
        for (int i=0; i<ks; i++)
            s += const_kernel[i]*Li[x + min(sy-1,max(0,y+i-ks2))*sx];
    }
    
    Lo[offset] = s;
}

static __global__ void medianFilter1D(const float *Li, float *Lo, int dir, int fsize, int sx, int sy)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (x >= sx || y >= sy)
        return;

    int i = 0;
    int ks = fsize/2;
    float v[MAX_MEDIAN];
    
    if (dir)
    {
        for (int xx = x - ks; xx <= x + ks; xx++)
            v[i++] = Li[xx+y*sx];
    }
    else
    {
        for (int yy = y - ks; yy <= y + ks; yy++)
            v[i++] = Li[x+yy*sx];
    }
    
    // bubble-sort
    float tmp;
    for (uint i=0; i<=ks; i++)
        for (uint j=i+1; j<fsize; j++)
            if (v[i] > v[j])
            {
                tmp = v[i];
                v[i] = v[j];
                v[j] = tmp;
            }

    // pick the middle one
    Lo[x + y*sx] = v[ks];
}

__global__ void mult(float *L, float m, const int sx, const int sy)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (x >= sx || y >= sy)
        return;

    int offset = x + y * sx;
    
    L[offset] *= m;
}

__global__ void resizeLanczos(float *Li, float *Lo, int sxi, int syi, int sxo, int syo)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (x >= sxo || y >= syo)
        return;
    
    float xp = x * float(sxi)/sxo,
          yp = y * float(syi)/syo;
    
    int x1 = floor(xp),
        y1 = floor(yp);
    
    float v = 0.0f, w = 0.0f, Lx, Ly, xx;
    int a = 3;
    for (int i=x1-a+1; i<=x1+a; i++)
    {
        xx = PI*(xp-i);
        Lx = a*sin(xx)*sin(xx/a)/(xx*xx); //L(x-i);
        if (Lx!=Lx) Lx = 1;
        for (int j=y1-a+1; j<=y1+a; j++)
        {
            xx = PI*(yp-j);
            Ly = a*sin(xx)*sin(xx/a)/(xx*xx);
            if (Ly!=Ly) Ly = 1;
            w += Lx*Ly;
            v += Li[min(sxi-1,max(0,i))+min(syi-1,max(0,j))*sxi]*Lx*Ly; //Li[i+j*sxi]
        }
    }
    
    Lo[x + y*sxo] = v/w;
}

__global__ void resizeNN(float *Li, float *Lo, int sxi, int syi, int sxo, int syo)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    unsigned int x1 = round(x * float(sxi)/sxo),
                 y1 = round(y * float(syi)/syo);
    
    if (x1 >= sxi || y1 >= syi)
        return;
    
    Lo[x + y*sxo] = Li[x1 + y1*sxi];
}

__global__ void convert(sint *Li, float *Lo, int sx, int sy)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (x >= sx || y >= sy)
        return;
    
    Lo[x + y*sx] = Li[x + y*sx];
}

extern "C"
{
void CUDA_filterConst(const float *Li, float *Lo,
                      int ks, int dir, int sx, int sy)
{
    dim3 blocks((sx+BLOCK_X-1)/BLOCK_X,(sy+BLOCK_Y-1)/BLOCK_Y);
    dim3 threads(BLOCK_X,BLOCK_Y);
    filterConst<<<blocks,threads>>>(Li, Lo, ks, dir, sx, sy);
    HANDLE_ERROR( cudaPeekAtLastError() );
    HANDLE_ERROR( cudaDeviceSynchronize() );
}

void CUDA_medianFilter1D(const float *Li, float *Lo, int dir, int fsize, int sx, int sy)
{
    dim3 blocks((sx+BLOCK_X-1)/BLOCK_X,(sy+BLOCK_Y-1)/BLOCK_Y);
    dim3 threads(BLOCK_X,BLOCK_Y);
    medianFilter1D<<<blocks,threads>>>(Li, Lo, dir, fsize, sx, sy);
    HANDLE_ERROR( cudaPeekAtLastError() );
    HANDLE_ERROR( cudaDeviceSynchronize() );
}

void CUDA_constructor(void** dev_data, int size)
{
    CUDA_destructor(dev_data);
    HANDLE_ERROR( cudaMalloc( dev_data, size) );
}

void CUDA_destructor(void** dev_data)
{
    if (*dev_data != NULL)
        HANDLE_ERROR( cudaFree( *dev_data ) );
}

void CUDA_copy(void* dataTo, void* dataFrom, int size, char type)
{
    cudaMemcpyKind kind;
    switch (type)
    {
        case HOST_TO_DEVICE:
            kind = cudaMemcpyHostToDevice;
            break;
        case DEVICE_TO_HOST:
            kind = cudaMemcpyDeviceToHost;
            break;
        case DEVICE_TO_DEVICE:
            kind = cudaMemcpyDeviceToDevice;
            break;
        case HOST_TO_HOST:
        default:
            kind = cudaMemcpyHostToHost;
            break;
    }

    HANDLE_ERROR( cudaMemcpy(dataTo, dataFrom, size, kind) );
}

void CUDA_mult(float *L, float m, int sx, int sy)
{
    dim3 blocks((sx+BLOCK_X-1)/BLOCK_X,(sy+BLOCK_Y-1)/BLOCK_Y);
    dim3 threads(BLOCK_X,BLOCK_Y);
    mult<<<blocks,threads>>>(L, m, sx, sy);
    HANDLE_ERROR( cudaPeekAtLastError() );
    HANDLE_ERROR( cudaDeviceSynchronize() );
}

void CUDA_resizeLanczos(float *Li, float *Lo, int sxi, int syi, int sxo, int syo)
{
    dim3 blocks((sxo+BLOCK_X-1)/BLOCK_X,(syo+BLOCK_Y-1)/BLOCK_Y);
    dim3 threads(BLOCK_X,BLOCK_Y);
    resizeLanczos<<<blocks,threads>>>(Li, Lo, sxi, syi, sxo, syo);
    HANDLE_ERROR( cudaPeekAtLastError() );
    HANDLE_ERROR( cudaDeviceSynchronize() );
}

void CUDA_resizeNN(float *Li, float *Lo, int sxi, int syi, int sxo, int syo)
{
    dim3 blocks((sxo+BLOCK_X-1)/BLOCK_X,(syo+BLOCK_Y-1)/BLOCK_Y);
    dim3 threads(BLOCK_X,BLOCK_Y);
    resizeNN<<<blocks,threads>>>(Li, Lo, sxi, syi, sxo, syo);
    HANDLE_ERROR( cudaPeekAtLastError() );
    HANDLE_ERROR( cudaDeviceSynchronize() );
}

void CUDA_convert(sint *Li, float *Lo, int sx, int sy)
{
    dim3 blocks((sx+BLOCK_X-1)/BLOCK_X,(sy+BLOCK_Y-1)/BLOCK_Y);
    dim3 threads(BLOCK_X,BLOCK_Y);
    convert<<<blocks,threads>>>(Li, Lo, sx, sy);
    HANDLE_ERROR( cudaPeekAtLastError() );
    HANDLE_ERROR( cudaDeviceSynchronize() );
}

void CUDA_copyToConst(float *kernel)
{
    HANDLE_ERROR( cudaMemcpyToSymbol( const_kernel, kernel, sizeof(float)*CONST_KERNEL_SIZE) );
}

}


// CudaTimer -------------------------------------------------------------------

CudaTimer::CudaTimer()
{
    HANDLE_ERROR(cudaEventCreate(&m_start));
    HANDLE_ERROR(cudaEventCreate(&m_end));
}

CudaTimer::~CudaTimer()
{
    HANDLE_ERROR(cudaEventDestroy(m_start));
    HANDLE_ERROR(cudaEventDestroy(m_end));
}

void CudaTimer::start()
{
    HANDLE_ERROR(cudaEventRecord(m_start, 0));
}

float CudaTimer::stop()
{
    HANDLE_ERROR(cudaEventRecord(m_end, 0));
    HANDLE_ERROR(cudaEventSynchronize(m_end));

    float elapsedTime;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, m_start, m_end));

    return elapsedTime;
}

#endif //UTIL_CUDA_CU




