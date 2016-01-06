
optix_prime_baking
==================

This sample shows how to precompute ambient occlusion with OptiX Prime, and store it on the
vertices of a mesh for use during final shading with OpenGL.  The steps are as follows:

  * **Distribute sample points uniformly over a mesh**. We place a minimum number of points per triangle, then use area-based sampling for any remaining points.  The total
  number of samples is a user parameter.

  * **Compute ambient occlusion at the sample points**.  We shoot batches of rays using OptiX Prime.  Each batch has a single jittered ray per sample point.

  * **Resample occlusion from sample points to vertices**.  If the external [Eigen 3](http://eigen.tuxfamily.org) template library was found during CMake configuration, then we use the 
  resampling method from "Least Squares Vertex Baking" (Kavan et al, EGSR 2011).  Eigen is open source.  In the absence of Eigen support, we use simpler barycentric resampling.  This shows more visual artifacts, especially when the input mesh has large triangles.

  * **Visualize occlusion in OpenGL as a vertex attribute**.

#### Requirements
  * A recent version of Visual Studio (tested with VS 2013)
  * CUDA 7.5+ and matching driver with supported GPU.
  * OptiX 3.9.0 from the shared_optix git repo (see below).  No separate install of OptiX is needed.

#### How to Build & Run

Quick build instructions for Visual Studio:

1) Clone the following nvpro-samples repositories:
  - //github.com/nvpro-samples/build_all.git
  - //github.com/nvpro-samples/shared_optix.git
  - //github.com/nvpro-samples/shared_sources.git
  - //github.com/nvpro-samples/optix_prime_baking.git

2) Download and install a recent version of [CMake](https://cmake.org)

3) Open CMake-gui, and generate Visual Studio projects:
  - Source code to: /nvpro_ samples/build_all
  - Build folder: /nvpro_samples/build
  - Click Configure and select a version of Visual Studio.
  - Click Generate

4) Open the nvpro_samples.sln into Visual Studio, and Build All

5) Select the optix_prime_baking sample as the Startup project

6) Click Run!

There are command line arguments for the input mesh, number of sample points, number of rays, etc.  Use the "-h" flag or
check main.cpp.
 
