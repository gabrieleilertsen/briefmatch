/**
 * \brief Cuda kernels for utility operations.
 *
 * Cuda GPU kernels for filtering, image resampling etc., as well as Cuda
 * memory management at timing measures.
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

#ifndef UTIL_CUDA_H
#define UTIL_CUDA_H

#include <stdio.h>
#include <string>
#include <curand.h>
#include <curand_kernel.h>

#define BLOCK_X 32
#define BLOCK_Y 32

#define MAX_MEDIAN 47

#define CONST_KERNEL_SIZE 100

#ifndef PI
#define PI 3.141592653589793115997963468544
#endif

#ifndef FLT_MAX
#define FLT_MAX 1e30
#endif

#ifndef FLT_MIN
#define FLT_MIN 1e-30
#endif

#ifndef sint
typedef short int sint;
#endif

#ifndef uchar
typedef unsigned char uchar;
#endif

#define HOST_TO_DEVICE 0
#define DEVICE_TO_HOST 1
#define DEVICE_TO_DEVICE 2
#define HOST_TO_HOST 3

class CudaException: public std::exception
{
  std::string msg;  
public:
    CudaException( std::string &message ) : msg(message) {}
    CudaException( const char *message ) { msg = std::string( message ); }

    ~CudaException() throw ()
    {
    }

    virtual const char* what() const throw()
    {
        return msg.c_str();
    }
};

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        char str[500];
        sprintf(str, "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        throw CudaException(str);
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

extern "C"
{
    void CUDA_constructor(void **dev_data, int size);
    void CUDA_destructor(void **dev_data);
    void CUDA_copy(void *dataTo, void *dataFrom, int size, char type);
    void CUDA_copyToConst(float *kernel);
    
    void CUDA_filterConst(const float *Li, float *Lo, int ks, int dir, int sx, int sy);
    void CUDA_medianFilter1D(const float *Li, float *Lo, int dir, int fsize, int sx, int sy);
    
    void CUDA_resizeLanczos(float *Li, float *Lo, int sxi, int syi, int sxo, int syo);
    void CUDA_resizeNN(float *Li, float *Lo, int sxi, int syi, int sxo, int syo);
    
    void CUDA_mult(float *L, float m, int sx, int sy);
    void CUDA_convert(sint *Li, float *Lo, int sx, int sy);
}

class CudaTimer
{
public:
    CudaTimer();
    ~CudaTimer();
    
    void start();
    float stop();
private:
    cudaEvent_t m_start, m_end;
};

#endif //UTIL_CUDA_H
