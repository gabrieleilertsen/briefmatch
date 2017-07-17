/**
 * \class OFPipeline
 *
 * \brief Optical flow pipeline using BriefMatch.
 *
 * OFPipeline setups an optical flow pipeline that can be executed. This
 * includes loading parameters, setting up memory structures, and then
 * executing the optical flow estimation. The execution loads images, performs
 * dense binary feature matching, filters the initial correspondence field, and
 * outputs the result to disc.
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

#ifndef OF_PIPELINE_H
#define OF_PIPELINE_H

#include <fstream>

#include "cuda.h"
#include "cuda_gl_interop.h"

#include "util_cuda.h"
#include <opencv2/opencv.hpp>
#include "briefmatch.h"
#include "arg_parser.h"

using namespace cv;

class OFPipeline
{
public:
    OFPipeline();
    ~OFPipeline();
    void clear();
    bool setParams(int argc, char* argv[]);
    void setup();
    bool processFrame();
    
private:
    bool hasExtension(const char *file_name, const char *extension);
    bool getFrameRange(std::string &frames, unsigned int &start, unsigned int &step, unsigned int &end);

    bool m_doWriteMOT, m_doWriteFLO, m_doWriteLDR,
         m_verbose, m_doPrintError;
    
    unsigned int m_endFrame, m_frame, m_stepFrame, m_startFrame,
                 m_sx, m_sy, m_sx_in, m_sy_in, m_c;

    std::string m_input, m_flowGT, m_output;
    
    CudaTimer *m_timer, *m_timerTot;
    
    BriefMatch *m_bm;
    BMParams *m_bmP;
    
    // GPU arrays
    float *dev_L, *dev_Lin, *dev_LPrev, *dev_Lin_prev;
};

#endif //OF_PIPELINE_H
