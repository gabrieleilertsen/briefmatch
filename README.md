# **BriefMatch**

## General
BriefMatch provides a fast GPU optical flow algorithm. The method performs a 
dense binary feature matching with [BRIEF descriptors](https://www.cs.ubc.ca/~lowe/525/papers/calonder_eccv10.pdf), 
and using the iterative propagation scheme from [PatchMatch](https://research.adobe.com/project/patchmatch/). 
The matching is followed by a trilateral filtering step that refines the 
correspondence field and removes outliers. All matching and filtering 
computations run on the GPU, which allows for real-time performance.

The method is described in our [SCIA2017 paper](http://vcl.itn.liu.se/publications/2017/EFU17/).
If you use the BriefMatch software for your research work, please consider
citing this as follows:

```
@InProceedings{EFU17,
  author       = "Eilertsen, Gabriel and Forssén, Per-Erik and Unger, Jonas",
  title        = "BriefMatch: Dense binary feature matching for real-time optical flow estimation",
  booktitle    = "Proceedings of the Scandinavian Conference on Image Analysis (SCIA17)",
  year         = "2017"
}
```

## Dependencies
The following libraries are required to compile BriefMatch:

 * [Cuda](https://developer.nvidia.com/cuda-toolkit) for GPU processing.
 * [OpenCV](http://opencv.org/) for image handling.

## Compilation and installation
Compilation is provided through CMake.

#### UNIX
For an out-of-tree build:

```
$ cd <path_to_briefmatch>
$ mkdir build
$ cd build
$ cmake ../
$ make
```

A set of advanced options can be provided with the CMake flag `ADVANCED_OPTIONS`:

```
$ cmake -DADVANCED_OPTIONS=1 ../
```

## BriefMatch usage
After compilation of BriefMatch, run `./briefmatch -h` to display available
input options. 

Quality/speed are best traded off by changing the up-sampling factor (`--up-sampling-x` and 
`--up-sampling-y`). Larger up-sampling gives more accurate flow vectors. Also,
with larger up-sampling there is more benefit in having longer
feature vectors (`--feature_length`). 

### Output format
Estimated optical flow fields can be output in three different formats:

  1. Binary floating point raw data, which first stores all x-components of 
     the flow, followed by the y-components. There is no meta data stored, so
     image size needs to be known in order to read the data.
  2. Middlebury .flo format, which also stores binary floating points. The
     format can be read in C++ and Matlab using the flowIO code provided at
     the [Middlebury benchmark webpage](http://vision.middlebury.edu/flow/data/).
  3. 8-bit color png images. The flow directions are encoded by color, and
     magnitudes by color saturation. The `--vis_max_motion` parameter can be
     used to scale this value to have maximum saturation and clamp all 
     values above.

### Examples
Following are three examples of different quality/speed trade-offs:

* Faster (lower quality):
  `./briefmatch --input data/RubberWhale/frame%02d.png --frames 7:14 --output output/flow%02d.png --up-sampling-x 1.5 --up-sampling-y 1.5 --feature_length 32`

* Moderate (medium quality):
  `./briefmatch --input data/RubberWhale/frame%02d.png --frames 7:14 --output output/flow%02d.png --up-sampling-x 2.9 --up-sampling-y 2.9 --feature_length 128`

* Slower (higher quality):
  `./briefmatch --input data/RubberWhale/frame%02d.png --frames 7:14 --output output/flow%02d.png --up-sampling-x 4.9 --up-sampling-y 4.9 --feature_length 512`

## Included libraries

BriefMatch utilizes the [flowIO](http://vision.middlebury.edu/flow/data/)
code provided on the [Middlebury benchmark website](http://vision.middlebury.edu/flow/). 
The code is used to colorcode 2D vectors for visualization of the
optical flow, and in order to store the optical flow using the flowIO format
(.flo).

FlowIO uses [imageLib](https://github.com/dscharstein/imageLib) in order to
handle images. Both flowIO and imageLib are included in `./util`. Please see
`./util/imageLib/` for further information and license information on imageLib.


## License

Copyright (c) 2017, The BriefMatch authors.
All rights reserved.

BriefMatch is distributed under a BSD license. See `LICENSE` for information.
