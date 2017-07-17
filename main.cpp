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

#include <iostream>

#include "of_pipeline.h"


int main(int argc, char *argv[])
{   
    // The optical flow pipeline
    OFPipeline of;

    try
    {
        // Read parameters from input
        if (!of.setParams(argc, argv))
            return 1;

        // Setup the pipeline
        of.setup();

        // Run pipeline
        std::cout << "\n\nRunning optical flow loop...\n";
        int status;
        do 
            status = of.processFrame(); 
        while (status);

        std::cout << "done!\n";
    }
    catch (ParserException &e)
    {
        fprintf(stderr, "\nbriefmatch input error: %s\n", e.what());
        return 1;
    }
    catch (BriefMatchException &e)
    {
        fprintf(stderr, "\nbriefmatch error: %s\n", e.what());
        return 1;
    }
    catch (CudaException &e)
    {
        fprintf(stderr, "\nCuda error: %s\n", e.what());
        return 1;
    }
    catch (std::exception & e)
    {
        fprintf(stderr, "\nError: %s\n", e.what());
        return 1;
    }
    
    return 0;
}



