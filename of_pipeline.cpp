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

#include "of_pipeline.h"
#include "config.h"

#include <sys/stat.h>

OFPipeline::OFPipeline()
{
    m_sx = m_sy = m_sx_in = m_sy_in = 0;

    m_doWriteMOT = 0;
    m_doWriteFLO = 0;
    m_doWriteLDR = 0;

    m_verbose = 0;
    m_doPrintError = 0;
    m_frame = -1;
    
    m_frame = 0; m_stepFrame = 1; m_endFrame = 0, m_startFrame = 0;
    m_timer = m_timerTot = NULL;
    m_bm = NULL; m_bmP = NULL;
    dev_L = dev_Lin = dev_LPrev = dev_Lin_prev = NULL;
}

OFPipeline::~OFPipeline()
{
    clear();
}

// Free host and device memory
void OFPipeline::clear()
{
    if (m_bm != NULL) delete m_bm;
    if (m_bmP != NULL) delete m_bmP;
    
    if (m_timer != NULL) delete m_timer;
    if (m_timerTot != NULL) delete m_timerTot;
    
    CUDA_destructor((void**)&dev_L);
    CUDA_destructor((void**)&dev_LPrev);
    CUDA_destructor((void**)&dev_Lin);
    CUDA_destructor((void**)&dev_Lin_prev);
    
    m_timer = m_timerTot = NULL;
    m_bm = NULL; m_bmP = NULL;
    dev_L = dev_Lin = dev_LPrev = dev_Lin_prev = NULL;
}

// Parse parameter options from command line
bool OFPipeline::setParams(int argc, char* argv[])
{
    m_bmP = new BMParams;
    std::string frames;
    float ux = 0, uy = 0;

    // Application usage info
    std::string info = std::string("briefmatch -- Computes the optical flow of a sequence of frames\n\n") +
                       std::string("Usage: briefmatch --input <frames> \\\n") +
                       std::string("                  --frames <start_frame:step:end_frame> \\\n") +
                       std::string("                  --output <output>\n");
    std::string postInfo = std::string("\nExample: briefmatch -i data/RubberWhale/frame%02d.png -f 7:14 -o output/mot_%02d.flo\n");
    ArgParser argHolder(info, postInfo);

    // Input arguments
    argHolder.add(&m_input,                  "--input",              "-i",   "Input sequence of frames", 0);
    argHolder.add(&m_output,                 "--output",             "-o",   "Output location of the estimated optical flow (.flo, .bin or .png)");
    argHolder.add(&frames,                   "--frames",             "-f",   "Input frames, formatted as startframe:step:endframe\n");
    argHolder.add(&m_bmP->maxmot,            "--vis-max-motion",     "-mm",  "Max-motion clamping, in conversion of flow field to color encoding", 0.0f, 1e10f);
    argHolder.add(&m_verbose,                "--verbose",            "-v",   "Verbose mode\n");
    argHolder.add(&m_bmP->N,                 "--feature-length",     "-fl",  "Length of BRIEF binary feature vectors", (unsigned int)(2), (unsigned int)(PATCH_MAX/4));
    argHolder.add(&m_bmP->patchRad,          "--patch-radius",       "-pr",  "BRIEF patch radius (in percent of image diagonal)", 0.0f, 1e10f);
    argHolder.add(&ux,                       "--up-sampling-x",      "-ux",  "Up-sampling factor for image width", 1.0f, 100.0f);
    argHolder.add(&uy,                       "--up-sampling-y",      "-uy",  "Up-sampling factor for image height\n", 1.0f, 100.0f);

#ifdef ADVANCED_OPTIONS
    argHolder.addInfo("Advanced options:\n");
    argHolder.add(&m_bmP->sigma,             "--sigma-downsampling", "-ss",  "Down-sampling filter size (gaussian filter, followed by NN sampling)", 0.0f, floor((CONST_KERNEL_SIZE - 1.0f)/2.0f)/3.0f);
    argHolder.add(&m_bmP->iterations,        "--iterations",         "-it",  "Number of iterations in neighbor search", (unsigned int)(0), (unsigned int)(1e2));
    argHolder.add(&m_bmP->jfMax,             "--jf-max",             "-jm",  "Max distance of jump flooding scheme (should be in multiples of 2)", (unsigned int)(2), (unsigned int)(1e4));
    argHolder.add(&m_bmP->fsize,             "--median-size",        "-ms",  "Median filter size in flow refinement filtering", (unsigned int)(0), (unsigned int)(MAX_MEDIAN));
    argHolder.add(&m_bmP->sEPE,              "--range-epe",          "-re",  "Sigma for correspondence field EPE filtering term", 0.0f, 1e10f);
    argHolder.add(&m_bmP->sI,                "--range-i",            "-ri",  "Sigma for input image filtering term", 0.0f, 1e10f);
    argHolder.add(&m_bmP->sS,                "--range-spatial",      "-rs",  "Sigma for spatial filering size\n", 0.0f, 1e10f);
    argHolder.add(&m_flowGT,                 "--flow-gt",            "-gt",  "Ground truth optical flow");
#endif

    // Parse arguments
    if (!argHolder.read(argc, argv))
        return 0;
    
    if (m_output.size() > 0)
    {
        // Check output format
        if (hasExtension(m_output.c_str(), ".flo"))
            m_doWriteFLO = true;
        else if (hasExtension(m_output.c_str(), ".bin"))
            m_doWriteMOT = true;
        else if (hasExtension(m_output.c_str(), ".png"))
            m_doWriteLDR = true;
        else
            throw ParserException("Unsupported output format. Supported formats are binary (.bin), Middlebury flow vector format (.flo) and color encoded images (.png).");
    }
    
    // Parse frame range
    if (frames.size() > 0 && !getFrameRange(frames, m_frame, m_stepFrame, m_endFrame))
        throw ParserException(std::string("Unable to parse frame range from '" + frames + "'. Valid format is startframe:step:endframe").c_str());
    
    // Valid frame range?
    if (m_endFrame < m_frame)
        throw ParserException(std::string("Invalid frame range '" + frames + "'. End frame should be >= start frame").c_str());

    m_startFrame = m_frame;
    m_c = 0;

    // Get image size
    char str[500];
    sprintf(str, m_input.c_str(), m_frame);    
    Mat I = imread(str, CV_LOAD_IMAGE_GRAYSCALE); //COLOR);
    if(!I.data )
        throw ParserException(std::string("Input image '" + std::string(str) + "' not found.").c_str());

    m_sx_in = I.cols; m_sy_in = I.rows;
    if (ux < 0.1f) ux = 2.740f;
    if (uy < 0.1f) uy = 3.052f;
    m_sx = ux*m_sx_in;
    m_sy = uy*m_sy_in;
    m_bmP->sx = m_sx;
    m_bmP->sy = m_sy;
    m_bmP->sx_in = m_sx_in;
    m_bmP->sy_in = m_sy_in;

    m_bmP->patchArea = 0.01f * m_bmP->patchRad * sqrt(m_sx*m_sx+m_sy*m_sy);

    if (m_flowGT.size() > 1) m_doPrintError = 1;

    return 1;
}

// Setup BriefMatch, allocate host and device memory, etc.
void OFPipeline::setup()
{
    fprintf(stderr, "I/O:\n");
    fprintf(stderr, "--------------------------------------------------------\n");
    fprintf(stderr, "Input:                   %s\n", m_input.c_str());
    fprintf(stderr, "Image resolution:        %dx%d\n", m_sx_in, m_sy_in);
    fprintf(stderr, "Frames:                  %d:%d:%d\n", m_frame, m_stepFrame, m_endFrame);
    fprintf(stderr, "Output:                  ");
    if (m_doWriteMOT || m_doWriteFLO || m_doWriteLDR)
        fprintf(stderr, "%s\n", m_output.c_str());
    else
        fprintf(stderr, "--\n");
    fprintf(stderr, "--------------------------------------------------------\n\n");

    // setup
    m_bm = new BriefMatch;
    m_bm->setVerbose(m_verbose);
    m_bm->setParams(*m_bmP);
    m_bm->setup();
    
    // for timing
    m_timer = new CudaTimer;
    m_timerTot = new CudaTimer;
    
    // allocate gpu image arrays
    CUDA_constructor((void**)&dev_Lin,      m_sx_in*m_sy_in*sizeof(float));
    CUDA_constructor((void**)&dev_Lin_prev, m_sx_in*m_sy_in*sizeof(float));
    CUDA_constructor((void**)&dev_L,        m_sx*m_sy*sizeof(float));
    CUDA_constructor((void**)&dev_LPrev,    m_sx*m_sy*sizeof(float));
    
    // gpu memory usage
    size_t avail;
    size_t total;
    cudaMemGetInfo( &avail, &total );
    size_t used = total - avail;
    printf("GPU memory status: %0.2fmb used, %0.2fmb available, %0.2fmb total\n", used/(1e6f), avail/(1e6f), total/(1e6f));
    
    m_timerTot->start();
}

// The main optical flow computation, estimating the flow between two frames
bool OFPipeline::processFrame()
{
    // All frames processed?
    if (m_frame>m_endFrame)
        return false;

    // Total elapsed time for one pair of frames
    float totTime = m_timerTot->stop();
    m_timerTot->start();

    char str[500], timings[1000];
    sprintf(str, m_input.c_str(), m_frame);

    if (m_verbose)
        printf("\n\n------------------------------------------------\nProcessing frame '%s'\n", str);
    else
    {
        unsigned int kv = 100*(m_frame-m_startFrame)/(m_endFrame-m_startFrame);
        if (kv >= m_c)
        {
            m_c += 5;
            printf("\t%3d%% (frame %3d in range [%d,%d])\n", kv, m_frame, m_startFrame, m_endFrame);
        }
    }
    
    // Read from disc
    m_timer->start();
    Mat I = imread(str, CV_LOAD_IMAGE_GRAYSCALE);
    if(!I.data )
        throw BriefMatchException(std::string("Input image '" + std::string(str) + "' not found.").c_str());
    I.convertTo(I, CV_32FC1);
    int pos = snprintf(timings, 500, "\tRead:\t\t\t%0.2fms", m_timer->stop());
    
    // Copy to GPU memory
    m_timer->start();
    CUDA_copy((void*)dev_Lin, (void*)(I.data), m_sx_in*m_sy_in*sizeof(float), HOST_TO_DEVICE);
    pos += snprintf(timings+pos, 500, "\n\tCopy to GPU:\t\t%0.2fms", m_timer->stop());
    
    // Up-sampling
    if (m_sx_in != m_sx || m_sy_in != m_sy)
    {
        if (m_verbose)
            printf("Up-sampling: [%d,%d] --> [%d,%d]\n", m_sx_in, m_sy_in, m_sx, m_sy);
        m_timer->start();
        CUDA_resizeLanczos(dev_Lin, dev_L, m_sx_in, m_sy_in, m_sx, m_sy);
        pos += snprintf(timings+pos, 500, "\n\tUp-sampling:\t\t%0.2fms", m_timer->stop());
    }
    else
        dev_L = dev_Lin;
    
    // Matching
    m_timer->start();
    m_bm->match(dev_LPrev, dev_L);
    pos += snprintf(timings+pos, 500, "\n\tMatching:\t\t%0.2fms", m_timer->stop());
    
    // Down-sampling
    m_timer->start();
    m_bm->downsample();
    pos += snprintf(timings+pos, 500, "\n\tDown-sampling:\t\t%0.2fms", m_timer->stop());
    
    // Flow refinement filtering
    m_timer->start();
    m_bm->process(dev_Lin_prev);
    pos += snprintf(timings+pos, 500, "\n\tFiltering:\t\t%0.2fms", m_timer->stop());
    
    // Frame processing time (without read/write)
    float procTime = m_timerTot->stop();
    
    // Swap current/previous frames, so the current frame is stored in processing of next frame
    float* tmp = dev_L; dev_L = dev_LPrev; dev_LPrev = tmp;
    tmp = dev_Lin; dev_Lin = dev_Lin_prev; dev_Lin_prev = tmp;
    m_bm->swap();
    
    // Output to disc
    if (m_doWriteMOT || m_doWriteFLO || m_doWriteLDR)
    {
        m_timer->start();

        // Download from GPU and convert
        m_bm->convert();
    
        // Flow vector color encoding
        if (m_doWriteLDR)
            m_bm->computeColors();
        
        char str[500];
        
        if (m_doWriteMOT) // write motion vectors to binary raw file
        {
            sprintf(str, m_output.c_str(), m_frame);
            m_bm->printRaw(str);
        }
        if (m_doWriteFLO) // write motion vectors to .flo file
        {
            sprintf(str, m_output.c_str(), m_frame);
            m_bm->print(str);
        }
        if (m_doWriteLDR) // write color encoded motion vectors
        {
            sprintf(str, m_output.c_str(), m_frame);
            m_bm->printVis(str);
        }

        pos += snprintf(timings+pos, 500, "\n\tWrite:\t\t\t%0.2fms", m_timer->stop());
    }
    
    // Error computation, if ground truth is provided
    if (m_doPrintError)
    {
        m_bm->convert();
        float err = m_bm->error(m_flowGT);
        printf("Error (EPE) = %f\n\n", err);
    }
    
    // Print timings
    if (m_verbose)
    {
        snprintf(timings+pos, 500, "\n\t--------\n\tProcessing time:\t%0.2fms / %0.2ffps (without read, write etc.)\n\tTotal time:\t\t%0.2fms / %0.2ffps",
                 procTime, 1000*1.0f/procTime, totTime, 1000*1.0f/totTime);
        printf("Timings:\n%s\n", timings);
    }

    m_frame += m_stepFrame;
    
    return true;
}


// Determine file extension
bool OFPipeline::hasExtension(const char *file_name, const char *extension)
{
    if( file_name == NULL )
        return false;
    size_t fn_len = strlen( file_name );
    size_t ex_len = strlen( extension );

    if( ex_len >= fn_len )
        return false;

    if( strcasecmp( file_name + fn_len - ex_len, extension ) == 0 )
        return true;  

    return false;
}

// Parse a frame range, given as <startFrame>:<step>:<endFrame>
bool OFPipeline::getFrameRange(std::string &frames, unsigned int &start, unsigned int &step, unsigned int &end)
{
    int nrDelim = -1;
    std::string::size_type pos = -1;
    
    do
    {
        pos = frames.find_first_of(":", pos+1);
        nrDelim++;
    } while (pos != std::string::npos);

    if (nrDelim < 1 || nrDelim > 2)
        return 0;
    
    unsigned int *range[3] = {&start, &step, &end};
    char *startPtr = &frames[0], *endPtr;
    
    for (size_t i=0; i<3; i+=3-nrDelim)
    {
        *(range[i]) = strtol(startPtr, &endPtr, 10);
        if (startPtr == endPtr)
            return 0;
        startPtr = endPtr + 1;
    }
    
    return 1;
}
