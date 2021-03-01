# Introduction

[Non-Local Means for Image Denoising using CUDA](report/report.pdf)

# Prerequisites

- Octave/Matlab
- NVIDIA GPU and CUDA

# Usage
``` shell
$ make <version>
```

If requirements are fulfilled then in the root directory:
``` shell
$ octave src/octave/pipeline.m <version name: v0 or v1 or v2> <args>
$ octave src/octave/pipeline.m v0 data/<image_file.jpg> <patch size>
$ octave src/octave/pipeline.m <v1 or v2> data/<image_file.jpg> <patch size> <grid size> <block size>
```
If requirements are not fulfilled, for CUDA versions *v1, v2* we can break the execution in three steps:
``` shell
octave src/octave/jpg2array.m <image_file.jpg> # create a noised image represented in a 2d array
```
In a compatible CUDA system (e.g. Google Colab):
``` shell
$ make <version: v1 or v2>
$ ./bin/<version> data/noise_image.txt <patch size> <grid size> <block size>
```
Rendering the filtered image:
``` shell
$ octave src/octave/rendering.m <version>
```

# Validation

In **early** stages of development, we need to be sure that our CPU implementation in C is working in the same way with a given tested Matlab implementation. So we had the following validation pipeline:

![validation](https://user-images.githubusercontent.com/6664730/109432641-c8b26080-7a14-11eb-9dd7-fb068f909ce3.png)

In the **current** state, we check only for CUDA versions, the filtered values with respect to CPU version. If requirements are fulfilled then for validation check, run the `pipeline.m` script, passing as a **last argument**: `--debug`.

Due to floating point arithmetic and the usage of *-use_fast_math* compiler flag, different values have been spotted. We assume valid a difference less than 1e-4. 
