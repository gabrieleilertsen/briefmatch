/**
 * \class OFPipeline
 *
 * \brief Per-pixel binary feature matching using BriefMatch.
 *
 * BriefMatch performs per-pixel matching between two images using binary
 * feature descriptors together with random search and propagation.
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

#ifndef BRIEF_MATCH_H
#define BRIEF_MATCH_H

#include "briefmatch_cuda.h"
#include "util_cuda.h"
#include "imageLib.h"

//#include "VclExr.h"
#include <opencv2/opencv.hpp>

#include <cstdio>
#include <math.h>
#include <iostream>
#include <exception>

typedef short int sint;

// Matching parameters
struct BMParams
{
    BMParams()
    {
        sx = sy = sx_in = sy_in = 64;
        iterations = 2;
        jfMax = 256;          // max jump flooding distance
        doubleProp = true;    // use 4 propagation directions? otherwise 2 is used
        filter = true;
        tInit = true;
        sigma = 0.5f;
        P = 11;
        fsize = 13;
        N = 128;              // length of binary feature vector
        patchArea = 75;
        sEPE = 0.4f;
        sI = 0.1f;
        sS = 6.0f;
        maxmot = 5.0f;
    }
    
    unsigned int sx, sy, sx_in, sy_in, N, iterations, jfMax, P, fsize;
    float patchArea, sigma, sEPE, sI, sS, maxmot;
    bool doubleProp, filter, tInit;
};

class BriefMatchException: public std::exception
{
  std::string msg;  
public:
    BriefMatchException( std::string &message ) : msg(message) {}
    BriefMatchException( const char *message ) { msg = std::string( message ); }

    ~BriefMatchException() throw ()
    {
    }

    virtual const char* what() const throw()
    {
        return msg.c_str();
    }
};

class BriefMatch 
{
public:
    BriefMatch();
    ~BriefMatch();
    
    void setup();
    void clean();
    
    void setParams(BMParams params) { m_params = params; }
    BMParams *getParams() { return &m_params; }
    
    void swap();
    void match(float *im1, float *im2);
    void downsample();
    void process(float *im1 = NULL);
    
    void convert();
    void print(std::string file);
    void computeColors(CFloatImage motim, CByteImage &colim);
    void computeColors();
    void printVis(std::string file);
    void printRaw(std::string file);
    float error(std::string filename);
    
    bool isSetup() { return m_setup; };
    void setVerbose(bool v) { m_verbose = v; };

private:
    float *dev_im1, *m_im1,
          *dev_im2, *m_im2;
    float *dev_dist;
    sint *dev_neighX,
         *dev_neighY;
    float *m_neighX, *dev_neighXF, *dev_neighXI, *dev_neighXI_m, *dev_neighXI_fin,
          *m_neighY, *dev_neighYF, *dev_neighYI, *dev_neighYI_m, *dev_neighYI_fin, *dev_tmp;
    unsigned int *dev_feat1, *dev_feat2;
    bool m_verbose, m_firstFrame;
    
    CFloatImage m_flow;
    CByteImage m_flowVis;
    
    curandState *dev_state;
    
    BMParams m_params;
    
    bool m_setup;
};

#endif //BRIEF_MATCH_H
