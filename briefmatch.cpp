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

#include "briefmatch.h"

#include <iostream>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>

#include "flowIO.h"
#include "colorcode.h"

BriefMatch::BriefMatch()
{
    m_setup = false;
    m_verbose = false;
    m_firstFrame = true;
    
    dev_im1 = dev_im2 = NULL;
    dev_dist = dev_tmp = NULL;
    dev_neighXF = dev_neighYF = dev_neighXI = dev_neighYI = NULL;
    dev_neighXI_fin = dev_neighYI_fin = dev_neighXI_m = dev_neighYI_m = NULL;
    dev_neighX = dev_neighY = NULL;
    dev_feat1 = dev_feat2 = NULL;
    dev_state = NULL;
}

BriefMatch::~BriefMatch()
{
    clean();
    
    delete[] m_neighX;
    delete[] m_neighY;
}

// Allocate cpu and gpu memory, and seed random field
void BriefMatch::setup()
{
    if (m_setup)
        return;

    // Exceptions if GPU constant memory pre-allocated arrays will be too small
    if (m_params.N > PATCH_MAX/4)
    {
        char m[500];
        sprintf(m, "Feature length exceeds maximum value. Current size is %d and maximum is %d.", m_params.N, PATCH_MAX/4); 
        throw BriefMatchException(m);
    }
    if (m_params.fsize > MAX_MEDIAN)
    {
        char m[500];
        sprintf(m, "Median filter size exceeds maximum value. Current size is %d and maximum is %d.", m_params.fsize, MAX_MEDIAN); 
        throw BriefMatchException(m);
    }
    if (ceil(m_params.sigma*3)*2 + 1 > CONST_KERNEL_SIZE)
    {
        char m[500];
        sprintf(m, "Maximum size of Gaussian down-sampling filter exceeded. Currently using sigma = %0.4f and maximum is %0.4f.",
                m_params.sigma, floor((CONST_KERNEL_SIZE - 1.0f)/2.0f)/3.0f); 
        throw BriefMatchException(m);
    }

    float h = sqrt(m_params.sx*m_params.sx+m_params.sy*m_params.sy);
    float h_in = sqrt(m_params.sx_in*m_params.sx_in+m_params.sy_in*m_params.sy_in);
    fprintf(stderr, "BriefMatch options:\n");
    fprintf(stderr, "--------------------------------------------------------\n");
    fprintf(stderr, "BRIEF settings:\n");
    fprintf(stderr, "\tFeature length:            %d\n", m_params.N);
    fprintf(stderr, "\tPatch radius:              %0.1f pixels (%0.4f%%)\n", m_params.patchArea, 100*m_params.patchArea/h);
    fprintf(stderr, "\tUp-sampling factor (x):    %0.4f\n", float(m_params.sx)/m_params.sx_in);
    fprintf(stderr, "\tUp-sampling factor (y):    %0.4f\n", float(m_params.sy)/m_params.sy_in);
    fprintf(stderr, "\tUp-sampling resolution:    %dx%d\n", m_params.sx, m_params.sy);
    fprintf(stderr, "\tDown-sampling sigma:       %0.4f pixels (%0.4f%%)\n", m_params.sigma, 100*m_params.sigma/h);
    fprintf(stderr, "\nCorrespondence field search:\n");
    fprintf(stderr, "\tIterations:                %d\n", m_params.iterations);
    fprintf(stderr, "\tJump flooding max:         %d\n", m_params.jfMax);
    fprintf(stderr, "\tTemporal init.:            ");
    if (m_params.tInit)
        fprintf(stderr, "Yes\n");
    else
        fprintf(stderr, "No\n");
    fprintf(stderr, "\nCross-trilateral flow refinement filtering:\n");
    fprintf(stderr, "\tMedian filter size:        %d pixels (%0.4f%%)\n", m_params.fsize, 100*m_params.fsize/h_in);
    fprintf(stderr, "\tSigma EPE:                 %0.4f\n", m_params.sEPE);
    fprintf(stderr, "\tSigma image:               %0.4f\n", m_params.sI);
    fprintf(stderr, "\tSigma spatial:             %0.4f pixels (%0.4f%%)\n", m_params.sS, 100*m_params.sS/h_in);
    fprintf(stderr, "--------------------------------------------------------\n\n");
    
    boost::mt19937 *rng = new boost::mt19937();
    rng->seed(time(NULL));

    boost::normal_distribution<> distribution(0, 0.2f);
    boost::variate_generator< boost::mt19937, boost::normal_distribution<> > randNorm(*rng, distribution);
    
    if (m_verbose)
    {
        printf("\n\nInitiating BRIEF matching.\n");
        printf("\tPatch sampling: ");
    }

    // BRIEF patch sampling
    float ma = -1, mi = 100, sm = 0;
    float *sampPatch = new float[4*m_params.N];
    for (unsigned int s=0; s<4*m_params.N; s++)
    {
        float val = randNorm();
        val = (2*float(val>0)-1)*pow(fabs(val), 2.0f);
        sampPatch[s] = m_params.patchArea*std::max(-1.0f, std::min(1.0f, val));
        ma = std::max(ma, sampPatch[s]);
        mi = std::min(mi, sampPatch[s]);
        sm += std::abs(sampPatch[s]);
        //printf("%f, ", sampPatch[s]);
    }
    if (m_verbose)
        printf("MAX = %f, MIN = %f, AVG = %f\n", ma, mi, sm/(4*m_params.N));
    
    // Gaussian filtering kernel
    float sig = m_params.sigma;
    m_params.P = ceil(sig*3)*2 + 1;
    int P2 = m_params.P/2;
    float *gaussKernel = new float[m_params.P];
    float w = 0;
    for (int i=-P2; i<=P2; i++)
    {
        gaussKernel[i+P2] = exp(-i*i/(2*sig*sig));
        w += gaussKernel[i+P2];
    }
    if (m_verbose)
        printf("\tGauss kernel (%d): ", m_params.P);
    for (unsigned int i=0; i<m_params.P; i++)
    {
        gaussKernel[i] /= w;
        if (m_verbose)
            printf("%f,", gaussKernel[i]);
    }
    if (m_verbose)
        printf("\n\n");
    
    int bs = sizeof(unsigned int)*8;
    int N_space = (m_params.N+bs-1)/bs;
    
    // host memory allocation
    m_neighX = new float[m_params.sx_in*m_params.sy_in];
    m_neighY = new float[m_params.sx_in*m_params.sy_in];
    m_flow.ReAllocate(CShape(m_params.sx_in, m_params.sy_in, 2));
    m_flowVis.ReAllocate(CShape(m_params.sx_in, m_params.sy_in, 3));
     
    // gpu memory allocation
    CUDA_constructor((void**)&dev_dist,    m_params.sx*m_params.sy*sizeof(float));
    CUDA_constructor((void**)&dev_neighX,  m_params.sx*m_params.sy*sizeof(sint));
    CUDA_constructor((void**)&dev_neighY,  m_params.sx*m_params.sy*sizeof(sint));
    CUDA_constructor((void**)&dev_neighXF, m_params.sx*m_params.sy*sizeof(float));
    CUDA_constructor((void**)&dev_neighYF, m_params.sx*m_params.sy*sizeof(float));
    CUDA_constructor((void**)&dev_tmp,     m_params.sx*m_params.sy*sizeof(float));
    
    CUDA_constructor((void**)&dev_feat1, m_params.sx*m_params.sy*N_space*sizeof(unsigned int));
    CUDA_constructor((void**)&dev_feat2, m_params.sx*m_params.sy*N_space*sizeof(unsigned int));
    
    CUDA_constructor((void**)&dev_neighXI,     m_params.sx_in*m_params.sy_in*sizeof(float));
    CUDA_constructor((void**)&dev_neighYI,     m_params.sx_in*m_params.sy_in*sizeof(float));
    CUDA_constructor((void**)&dev_neighXI_m,   m_params.sx_in*m_params.sy_in*sizeof(float));
    CUDA_constructor((void**)&dev_neighYI_m,   m_params.sx_in*m_params.sy_in*sizeof(float));
    CUDA_constructor((void**)&dev_neighXI_fin, m_params.sx_in*m_params.sy_in*sizeof(float));
    CUDA_constructor((void**)&dev_neighYI_fin, m_params.sx_in*m_params.sy_in*sizeof(float));
    
    CUDA_constructor((void**)&dev_state,  m_params.sx*m_params.sy*sizeof(curandState));
    CUDA_initRandom(dev_state, m_params.sx, m_params.sy);
    
    CUDA_copyPatchToConst(sampPatch);
    CUDA_copyToConst(gaussKernel);
    
    delete[] sampPatch;
    delete[] gaussKernel;
    
    m_setup = true;

    if (m_verbose)
        printf("BRIEF matching setup complete.\n");
}

// Free GPU memory
void BriefMatch::clean()
{
    if (m_setup)
    {
        CUDA_destructor((void**)&dev_dist);
        CUDA_destructor((void**)&dev_neighX);
        CUDA_destructor((void**)&dev_neighY);
        CUDA_destructor((void**)&dev_neighXF);
        CUDA_destructor((void**)&dev_neighYF);
        CUDA_destructor((void**)&dev_neighXI);
        CUDA_destructor((void**)&dev_neighYI);
        CUDA_destructor((void**)&dev_neighXI_m);
        CUDA_destructor((void**)&dev_neighYI_m);
        CUDA_destructor((void**)&dev_neighXI_fin);
        CUDA_destructor((void**)&dev_neighYI_fin);
        CUDA_destructor((void**)&dev_tmp);
        
        CUDA_destructor((void**)&dev_feat1);
        CUDA_destructor((void**)&dev_feat2);
        CUDA_destructor((void**)&dev_state);
    }
}

void BriefMatch::swap()
{
    unsigned int *tmp = dev_feat1;
    dev_feat1 = dev_feat2;
    dev_feat2 = tmp;
}

// The matching algorithm
void BriefMatch::match(float *im1, float *im2)
{
    if (!m_setup)
        throw BriefMatchException("BriefMatch is not initialized before use.");
    
    dev_im1 = im1;
    dev_im2 = im2;
    
    int bs = sizeof(unsigned int)*8;
    int N_space = (m_params.N+bs-1)/bs;

    // Calculate feature vectors
    CUDA_binaryFeatures(dev_im2, dev_feat2,
                        m_params.sx, m_params.sy,
                        m_params.N, N_space);
    
    // Initialize neighbors and distances
    CUDA_initialize(dev_im1, dev_im2, dev_feat1, dev_feat2,
                    dev_dist, dev_neighX, dev_neighY,
                    m_params.sx, m_params.sy,
                    m_firstFrame || !m_params.tInit, m_params.N, N_space);
    
    int pd = -1;
    float dist = sqrt(ceil(m_params.sx*0.5f/3.0f));
    
    // Iterate propagation and random search
    for (unsigned int i=0; i<m_params.iterations; i++)
    {
        // Propagation of neighbors
        CUDA_propagate(dev_im1, dev_im2, dev_feat1, dev_feat2,
                       dev_dist, dev_neighX, dev_neighY,
                       m_params.sx, m_params.sy,
                       pd, m_params.jfMax, m_params.doubleProp,
                       m_params.N, N_space);

        // Random neighbor search
        CUDA_randomSearch(dev_im1, dev_im2, dev_feat1, dev_feat2,
                          dev_dist, dev_neighX, dev_neighY,
                          dev_state, m_params.sx, m_params.sy, dist,
                          m_params.N, N_space);
        
        pd = -pd;
    }

    m_firstFrame = false;
}

// Down-sampling of flow field
void BriefMatch::downsample()
{
    CUDA_convert(dev_neighX, dev_neighXF, m_params.sx, m_params.sy);
    CUDA_convert(dev_neighY, dev_neighYF, m_params.sx, m_params.sy);
    
    if (m_params.sx_in != m_params.sx || m_params.sy_in != m_params.sy)
    {
        // Separable Gaussian filter of flow x-components
        CUDA_filterConst(dev_neighXF, dev_tmp, m_params.P, 1, m_params.sx, m_params.sy);
        CUDA_filterConst(dev_tmp, dev_neighXF, m_params.P, 0, m_params.sx, m_params.sy);
        
        // Separable Gaussian filter of flow x-components
        CUDA_filterConst(dev_neighYF, dev_tmp, m_params.P, 1, m_params.sx, m_params.sy);
        CUDA_filterConst(dev_tmp, dev_neighYF, m_params.P, 0, m_params.sx, m_params.sy);
        
        // Nearest neighbor down-sampling of filtered flow
        CUDA_resizeNN(dev_neighXF, dev_neighXI, m_params.sx, m_params.sy, m_params.sx_in, m_params.sy_in);
        CUDA_resizeNN(dev_neighYF, dev_neighYI, m_params.sx, m_params.sy, m_params.sx_in, m_params.sy_in);
        
        // Scale by input size/up-sampling size
        CUDA_mult(dev_neighXI, float(m_params.sx_in)/m_params.sx, m_params.sx_in, m_params.sy_in);
        CUDA_mult(dev_neighYI, float(m_params.sy_in)/m_params.sy, m_params.sx_in, m_params.sy_in);
    }
    else
    {
        dev_neighXI = dev_neighXF;
        dev_neighYI = dev_neighYF;
    }
}

// Flow refinement filtering
void BriefMatch::process(float *im1)
{
    if (m_params.filter)
    {
        // Separable median filtering of flow x-component
        CUDA_medianFilter1D(dev_neighXI, dev_tmp,   1, m_params.fsize, m_params.sx_in, m_params.sy_in);
        CUDA_medianFilter1D(dev_tmp, dev_neighXI_m, 0, m_params.fsize, m_params.sx_in, m_params.sy_in);
        
        // Separable median filtering of flow y-component
        CUDA_medianFilter1D(dev_neighYI, dev_tmp,   1, m_params.fsize, m_params.sx_in, m_params.sy_in);
        CUDA_medianFilter1D(dev_tmp, dev_neighYI_m, 0, m_params.fsize, m_params.sx_in, m_params.sy_in);
        
        // Trilateral filter
        CUDA_cbFilter(dev_neighXI, dev_neighYI,
                      dev_neighXI_fin, dev_neighYI_fin,
                      dev_neighXI_m, dev_neighYI_m,
                      im1,
                      m_params.sEPE*m_params.sEPE, m_params.sI*m_params.sI, m_params.sS*m_params.sS,
                      m_params.sx_in, m_params.sy_in);
    }
    else
    {
        dev_neighXI_fin = dev_neighXI;
        dev_neighYI_fin = dev_neighYI;
    }
}

// Download from GPU and convert to CFloatImage
void BriefMatch::convert()
{
    CUDA_copy((void*)m_neighX, (void*)dev_neighXI_fin, m_params.sx_in*m_params.sy_in*sizeof(float), DEVICE_TO_HOST);
    CUDA_copy((void*)m_neighY, (void*)dev_neighYI_fin, m_params.sx_in*m_params.sy_in*sizeof(float), DEVICE_TO_HOST);
    
    for (unsigned int y = 0; y < m_params.sy_in; y++)
    {
	    for (unsigned int x = 0; x < m_params.sx_in; x++)
	    {
	        float *pix = &m_flow.Pixel(x, y, 0);
	        
	        pix[0] = m_neighX[x+y*m_params.sx_in];
	        pix[1] = m_neighY[x+y*m_params.sx_in];
	    }
    }
}

// Flow vector color encoding
void BriefMatch::computeColors(CFloatImage motim, CByteImage &colim)
{
    unsigned int x, y;
    // determine motion range:
    float maxx = -999, maxy = -999;
    float minx =  999, miny =  999;
    float maxrad = -1;
    for (y = 0; y < m_params.sy_in; y++) {
	    for (x = 0; x < m_params.sx_in; x++) {
	        float fx = motim.Pixel(x, y, 0);
	        float fy = motim.Pixel(x, y, 1);
	        if (unknown_flow(fx, fy))
		        continue;
	        maxx = __max(maxx, fx);
	        maxy = __max(maxy, fy);
	        minx = __min(minx, fx);
	        miny = __min(miny, fy);
	        float rad = sqrt(fx * fx + fy * fy);
	        maxrad = __max(maxrad, rad);
	    }
    }
    if (m_verbose)
        printf("max motion: %.4f  motion range: u = %.3f .. %.3f;  v = %.3f .. %.3f\n",
	       maxrad, minx, maxx, miny, maxy);


    if (m_params.maxmot > 0)
	    maxrad = m_params.maxmot;

    if (maxrad == 0)
	    maxrad = 1;

	//fprintf(stderr, "normalizing by %g\n", maxrad);

    for (y = 0; y < m_params.sy_in; y++) {
	    for (x = 0; x < m_params.sx_in; x++)
	    {
	        float fx = motim.Pixel(x, y, 0);
	        float fy = motim.Pixel(x, y, 1);
	        uchar *pix = &colim.Pixel(x, y, 0);
	        if (unknown_flow(fx, fy))
	        {
		        pix[0] = pix[1] = pix[2] = 0;
	        } 
	        else
		        computeColor(fx/maxrad, fy/maxrad, pix);
	    }
    }
}

void BriefMatch::computeColors()
{
    computeColors(m_flow, m_flowVis);
}

// Print color encoded flow to file
void BriefMatch::printVis(std::string file)
{
    try
    {
        WriteImageVerb(m_flowVis, file.c_str(), m_verbose);
    }
    catch (CError &e)
    {
        throw BriefMatchException(std::string("Cannot write to image file.\n\t" + std::string(e.message)).c_str());
    }
}

// Print binary raw flow data
void BriefMatch::printRaw(std::string file)
{
    FILE* pFile;
    pFile = fopen(file.c_str(), "wb");
    if (!pFile)
        throw BriefMatchException(std::string("Invalid output path. Cannot write to file '" + file + "'.").c_str());

    fwrite(m_neighX, sizeof(float), m_params.sx_in*m_params.sy_in, pFile);
    fwrite(m_neighY, sizeof(float), m_params.sx_in*m_params.sy_in, pFile);
    fclose(pFile);
}

// Print flow to .flo file
void BriefMatch::print(std::string file)
{
    try
    {
        WriteFlowFile(m_flow, file.c_str());
    }
    catch (CError &e)
    {
        throw BriefMatchException(std::string("Cannot write .flo file.\n\t" + std::string(e.message)).c_str());
    }
}

// Compute end-point error (EPE), compared to ground truth flow
float BriefMatch::error(std::string file)
{
    CFloatImage I_gt;
    float err = 0;

    try
    {
        ReadFlowFile(I_gt, file.c_str());
        
        CShape sh = I_gt.Shape();
        int width = sh.width, height = sh.height;
        
        int p = 0;
        float e, fx, fy, dx, dy;
        int x, y, N = 0;
        for (y = p; y < height-p; y++)
        {
    	    for (x = p; x < width+p; x++)
    	    {
    	        fx = I_gt.Pixel(x, y, 0);
    	        fy = I_gt.Pixel(x, y, 1);
    	        if (unknown_flow(fx, fy))
    		        continue;
    		    
    	        dx = fx-m_flow.Pixel(x, y, 0); //m_neighX[x+y*m_params.sx_in];
    	        dy = fy-m_flow.Pixel(x, y, 1); //m_neighY[x+y*m_params.sx_in];
    	        
    	        e = sqrt(dx * dx + dy * dy);
    	        err += e;
    	        N++;
    	    }
        }
        
        err /= N;
        
        CByteImage colim;
        colim.ReAllocate(CShape(m_params.sx_in, m_params.sy_in, 3));
        computeColors(I_gt, colim);
        //WriteImageVerb(colim, "output/gt.png", 0);
    }
    catch (CError &e)
    {
        throw BriefMatchException(std::string("Failure in flow error computation:\n\t" + std::string(e.message)).c_str());
    }
    
    return err;
}


