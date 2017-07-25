cmake_minimum_required (VERSION 2.8)
project (BriefMatch)

option(ADVANCED_OPTIONS
  "Include advanced options" 0)

option(USE_OPENCV
  "Use OpenCV for a more flexible and faster image input" 1)

if (APPLE)
    set(CMAKE_MACOSX_RPATH 1)
    set(BUILD_WITH_INSTALL_RPATH 1)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "10.8")
endif ()

if (CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall -Wno-long-long -pedantic")
endif ()

# === Search for dependencies ==================================================
find_package(CUDA REQUIRED)
find_package(PNG REQUIRED)
if (USE_OPENCV)
    find_package(OpenCV)
endif ()

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
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -Xcompiler -fPIC -Wno-deprecated-gpu-targets -use_fast_math" )
    else()
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -Wno-deprecated-gpu-targets -use_fast_math" )
    endif()
endif( NOT CUDA_FOUND )

# === libPNG ==================================================================
message( " * libPNG" )
if( NOT PNG_FOUND )
    message( WARNING "libPNG not found." )
else( NOT PNG_FOUND )
    message( "\tlibPNG version: ${PNG_VERSION_STRING}" )
    message( "\tinclude: ${PNG_INCLUDE_DIRS}\n\tlibs: ${PNG_LIBRARIES}" )
endif( NOT PNG_FOUND )

# === OpenCV ==================================================================
if (USE_OPENCV)
    message( " * OpenCV" )
    if( NOT OpenCV_FOUND )
        message( WARNING "OpenCV not found." )
        message( "\tBriefMatch will be compiled without OpenCV support." )
        set( USE_OPENCV 0 )
    else( NOT OpenCV_FOUND )
        message( "\tinclude: ${OpenCV_INCLUDE_DIRS}\n\tlibs: ${OpenCV_LIBS}" )
    endif( NOT OpenCV_FOUND )
endif ()

message( "==============================================================\n" )

# === Target ==================================================================
include_directories(
    "${PROJECT_SOURCE_DIR}/kernels"
    "${PROJECT_SOURCE_DIR}/util"
    "${PROJECT_SOURCE_DIR}/util/imageLib"
    "${PROJECT_BINARY_DIR}"
    "${CUDA_INCLUDE_DIRS}"
    "${PNG_INCLUDE_DIRS}"
)

if (USE_OPENCV)
    include_directories(
        "${OpenCV_INCLUDE_DIRS}"
    )
endif ()

target_link_libraries(image_lib ${PNG_LIBRARIES})

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

if (USE_OPENCV)
    target_link_libraries(briefmatch image_lib ${CUDA_LIBRARIES} ${PNG_LIBRARIES} ${OpenCV_LIBS})
else  (USE_OPENCV)
    target_link_libraries(briefmatch image_lib ${CUDA_LIBRARIES} ${PNG_LIBRARIES})
endif ()


# === Config ===================================================================
configure_file(
  "${PROJECT_SOURCE_DIR}/config.h.in"
  "${PROJECT_BINARY_DIR}/config.h"
)


