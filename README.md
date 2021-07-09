# The CUDA Coarsening Compiler

The CUDA Coarsening Compiler (CCC) is an automatic implementation of thread coarsening using CUDA and LLVM.

## Overview

### What is thread coarsening and how does it work?

Thread coarsening is loop unrolling applied to GPU kernels - kernel unrolling. GPU kernels exist within a implied execution loops over thread blocks and threads. Thread coarsening unrolls one (or several) of these by a *coarsening factor*, applying code duplication to the kernel body and reducing the iteration space accordingly.

### How does the compiler work?

The coarsening compilation happens in two stages. First, the device code is coarsened by applying code duplication to GPU kernel bodies, with respect to a set of coarsening parameters. Second, the host code is checked for kernel invocations, and the iteration space is reduced accordingly. Both are implemented as LLVM passes.

The CUDA Coarsening Compiler can operate in `static` and `dynamic` mode. Static mode coarsens a given GPU kernel with respect to one specific configuration. Dynamic mode, on the other hand, creates a whole bunch of differently coarsened kernels and links them all into the binary. You almost always want to be using dynamic mode (which is the default), as it allows you to select the most suitable kernel config at runtime, once the problem size is known.

## Building for dev
Building can be done by simply invoking `cmake` followed by `make`, which should be sufficient when working on CCC itself.

## Building for external use
In order to coarsen programs of your own projects, it is recommended that you add CCC as a cmake dependency. This section will walk you through what to do. First, it is recommended that you build CCC the following way:

```
cmake /path/to/cuda-coarsening-compiler -DCMAKE_INSTALL_PREFIX=/path/to/install-dir"
make install
```

The `CMAKE_INSTALL_PREFIX` can be omitted, but is useful for installing into user space.

Next, set up your own `CMakeLists.txt` as follows:
```
project(MyProject LANGUAGES C CXX CUDA)

cmake_minimum_required(VERSION 3.17)
find_package(cuda-coarsening-compiler REQUIRED)

coarsening_compile(TARGET my_program SOURCES my_program.cu)
```

You can add any include directories in the usual way before invoking `coarsening_compile()`, which will then perform the compilation for you.

The following arguments can be provided to the build system:
* `CCC_DEVICE_ARCH` expects a CUDA device architecture, for example `sm_52`
* `CCC_COMPUTE_ARCH` expects a CUDA compute architecture, for instance `compute_52`A
* `OPT` expects any optimization flags. Defaults to `-O3` and should not be changed.
* `CUDA_PATH` is used to locate CUDA_PATH/lib64. Defaults to `/usr/local/cuda`.

It's now time to build your project. If you have specified a `CMAKE_INSTALL_PREFIX` path before, you will need to specify a `CMAKE_PREFIX_PATH` (note the appended `/lib`).
```
cmake /path/to/my-project -DCMAKE_PREFIX_PATH=/path/to/install-dir/lib -DCCC_DEVICE_ARCH=sm_52 -DCCC_COMPUTE_ARCH=compute_52
make
```

Although dynamic mode is recommended, the `coarsening_compile()` function also supports static mode compilation. Here is the full definition:
```
coarsening_compile(TARGET target
                   SOURCES src1 [src2...]
                   [STATIC
                    KERNEL <specific-kernel-name|all>
                    MODE   <thread|block|dynamic>
                    DIMENSION <x|y|z>
                    STRIDE <number>
                   ])
```

## Coarsening in-tree
In order to apply coarsening to kernels within this project, uncomment the following line to `CMakeLists.txt`:
```
include("${CMAKE_SOURCE_DIR}/cmake/cuda-coarsening-compiler-config.cmake.in")
```
Adjust your build commands as described above.

## Build environment

The `coarsening_compile()` CMake function requires tools from both LLVM and CUDA in order to work, which need to be visible in your `$PATH`:

* From LLVM: `opt`, `llc`, `clang` and `clang++`.
* From CUDA: `ptxas` and `fatbinary`.

The project was tested with LLVM 9.0.1 and CUDA 10.1.

## Running a coarsened programm

By default, the CUDA Coarsening Compiler operates in `dynamic` mode. This means that a bunch of differently coarsened kernels are compiled and linked into the target binary. By setting the `RPC_CONFIG` environment variable, you can then select at runtime which version of the kernel you would like to execute, although the choice can be made automatically in the future.
The `RPC_CONFIG` environment variable expects the following format: `<kernelname>,<dim>,<block/thread>,<factor>,<stride>`. For instance, a dynamically coarsened program can be invoked as follows:
```RPC_CONFIG=my_kernel,x,block,4,1 ./my-program```

