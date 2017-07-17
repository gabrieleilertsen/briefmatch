cmake_minimum_required (VERSION 2.8)
project (BriefMatch)

option(ADVANCED_OPTIONS
  "Include advanced options" 0)

set( CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake )

if(APPLE)
    set(CMAKE_MACOSX_RPATH 1)
    set(BUILD_WITH_INSTALL_RPATH 1)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "10.8")
endif()

if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall -Wno-long-long -pedantic")
endif()

# === Search for dependencies ==================================================
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

# === Add imageLib library ===================================================
add_library(image_lib SHARED
    ${PROJECT_SOURCE_DIR}/util/imageLib/Convert.cpp
    ${PROJECT_SOURCE_DIR}/util/imageLib/Image.cpp
    ${PROJECT_SOURCE_DIR}/util/imageLib/ImageIO.cpp
    ${PROJECT_SOURCE_DIR}/util/imageLib/ImageIOpng.cpp
    ${PROJECT_SOURCE_DIR}/util/imageLib/RefCntMem.cpp
)

message( "\n\n==============================================================" )
message( "Dependencies:\n" )

# === Cuda ==================================================================
message( " * Cuda" )
if( NOT CUDA_FOUND )
    message( WARNING "Cuda not found." )
else( NOT CUDA_FOUND )
    message( "\tCuda version: ${CUDA_VERSION}" )
    message( "\tinclude: ${CUDA_INCLUDE_DIRS}\n\tlibs: ${CUDA_LIBRARIES}" )

    # NVCC options
    set(CUDA_PROPAGATE_HOST_FLAGS OFF)
    set(CUDA_NVCC_FLAGS
        ${CUDA_NVCC_FLAGS};
        -Xcompiler -fPIC -Wno-deprecated-gpu-targets
    )
endif( NOT CUDA_FOUND )

# === OpenCV ==================================================================
message( " * OpenCV" )
if( NOT OpenCV_FOUND )
    message( WARNING "OpenCV not found." )
else( NOT OpenCV_FOUND )
    message( "\tinclude: ${OpenCV_INCLUDE_DIRS}\n\tlibs: ${OpenCV_LIBS}" )
endif( NOT OpenCV_FOUND )

message( "==============================================================\n" )

# === Target ==================================================================
include_directories ("${PROJECT_SOURCE_DIR}/kernels" "${PROJECT_SOURCE_DIR}/util" "${PROJECT_SOURCE_DIR}/util/imageLib" "${PROJECT_BINARY_DIR}" "${CUDA_INCLUDE_DIRS}" "${OpenCV_INCLUDE_DIRS}")

target_link_libraries(image_lib)

cuda_add_executable(briefmatch 
    ${PROJECT_SOURCE_DIR}/main.cpp
    ${PROJECT_SOURCE_DIR}/of_pipeline.cpp
    ${PROJECT_SOURCE_DIR}/briefmatch.cpp
    ${PROJECT_SOURCE_DIR}/kernels/briefmatch_cuda.cu
    ${PROJECT_SOURCE_DIR}/kernels/util_cuda.cu
    ${PROJECT_SOURCE_DIR}/util/arg_parser.cpp
    ${PROJECT_SOURCE_DIR}/util/colorcode.cpp
    ${PROJECT_SOURCE_DIR}/util/flowIO.cpp
)

target_link_libraries(briefmatch image_lib ${CUDA_LIBRARIES} ${OpenCV_LIBS})


# === Config ===================================================================
configure_file (
  "${PROJECT_SOURCE_DIR}/config.h.in"
  "${PROJECT_BINARY_DIR}/config.h"
)


