
optix_prime_baking
==================

![Rocket Sled](https://github.com/nvpro-samples/optix_prime_baking/blob/master/doc/sled_multiple_meshes.png)

This sample shows how to precompute ambient occlusion with OptiX Prime, and store it on the
vertices of a mesh for use during final shading with OpenGL.  The steps are as follows:

  * **Distribute sample points over a mesh**. We place a minimum number of points per triangle, then use area-based sampling for any remaining points.  The total number of samples is a user parameter.

  * **Compute ambient occlusion at the sample points**.  To help limit memory usage, we shoot rays in multiple batches using OptiX Prime.  Each batch has a single jittered ray on a subset of the sample points.  Geometry can be instanced and/or marked as a *blocker* which occludes rays but does not receive sample points of its own.

  * **Resample occlusion from sample points to vertices**.  If the external [Eigen 3](http://eigen.tuxfamily.org) template library was found during CMake configuration, then we use the 
  filtering method from "Least Squares Vertex Baking" (Kavan et al, EGSR 2011).  Eigen is open source.  In the absence of Eigen support, we use simpler barycentric resampling.  This shows more visual artifacts, especially when the input mesh has large triangles.  A copy of Eigen is included in the "eigen" subdirectory and will be used by default.

  * **Visualize occlusion in OpenGL as a vertex attribute**.

#### Requirements
  * A recent version of Visual Studio (tested with VS 2013 on Windows 7) or gcc (tested with gcc 4.8.4 on Ubuntu 14.04) 
  * CUDA 7.5+ and matching driver with supported GPU.
  * A recent version of CMake (tested with 2.8.12).
  * OptiX 3.9.0 from the shared_optix git repo (see below).  No separate install of OptiX is needed.

#### How to Build & Run

Quick build instructions:

1) Clone the following nvpro-samples repositories:
  - //github.com/nvpro-samples/build_all.git
  - //github.com/nvpro-samples/shared_optix.git
  - //github.com/nvpro-samples/shared_sources.git
  - //github.com/nvpro-samples/shared_external.git
  - //github.com/nvpro-samples/optix_prime_baking.git

2) Download and install a recent version of [CMake](https://cmake.org)

3) Open CMake-gui (Windows) or ccmake (Linux):
  - Source code to: /nvpro_ samples/build_all
  - Build folder: /nvpro_samples/build_all/build
  - Optional: toggle MODELS_DOWNLOAD_ENABLE to download extra scenes during the configure step.
  - Configure and select a compiler if prompted.
  - Generate

4) Open the nvpro_samples.sln into Visual Studio, and Build All.  On Linux, 'make' in the build directory.

5) Select the optix_prime_baking sample as the Startup project in VS.

6) Click Run in VS, or run the 'nvpro_samples/bin_x64/optix_prime_baking' binary in Linux.

The sample is configured on the command line; use the "-h" flag to list options or check main.cpp.  The options at the time the sample was created are shown below:
~~~
App options:
  -h  | --help                          Print this usage message
  -f  | --file <scene_file>             Specify model to be rendered (obj, bk3d, or bk3d.gz).
  -i  | --instances <n>                 Number of instances per mesh (default 1).  For testing.
  -r  | --rays    <n>                   Number of rays per sample point for gather (default 64)
  -s  | --samples <n>                   Number of sample points on mesh (default 3 per face; any extra samples are based on area)
  -t  | --samples_per_face <n>          Minimum number of samples per face (default 3)
        --no_ground_plane               Disable virtual XZ ground plane
  -w  | --regularization_weight <w>     Regularization weight for least squares, positive range. (default 0.1)
        --no_least_squares              Disable least squares filtering
 ~~~
 
#### Least Squares Filtering

The effect of filtering on a simplified Lucy model (80k faces) is shown below.  The right image uses least squares filtering (regularization weight 0.1) and is noticeably smoother when the triangles are this large.
![Lucy Image](https://github.com/nvpro-samples/optix_prime_baking/blob/master/doc/lucy_least_squares_comparison.png)

Mesh detail showing triangle size: 
![Lucy mesh detail](https://github.com/nvpro-samples/optix_prime_baking/blob/master/doc/lucy_meshlab.png)

#### Instancing

All geometry consists of *instances*, which are pairs of meshes and transforms.  A mesh referenced by multiple instances is only stored once in memory.  For example, the scene below has 3 instances of a mesh with 1.3M triangles, and was baked using less than 1 GB of GPU memory for the scene, rays, etc.  This scene also has an invisible ground plane marked as a *blocker*.

![Instancing Example](https://github.com/nvpro-samples/optix_prime_baking/blob/master/doc/hunter_instances.png)

#### Supported scene formats 

Loaders are provided for OBJ and [Bak3d](https://github.com/tlorach/Bak3d).  The OBJ loader flattens all groups into a single mesh.  The bk3d loader preserves separate meshes, as shown in the teaser image above of a rocket sled with 109 meshes.  Use the utilities in the Bak3d repo to convert other formats, or write a new loader for your favorite format and add it to the "loaders" subdirectory.

The rocket sled .bk3d file can be downloaded via MODELS_DOWNLOAD_ENABLE in the cmake config.

#### Performance

Timings for the sled scene on an NVIDIA Quadro M6000 GPU are shown below, including the optional least squares filtering step which is not GPU accelerated.

~~~
> ../../../bin_x64/optix_prime_baking -f ../assets/sled_v134.bk3d.gz -w 1
Load scene ...                 96.27 ms
Loaded scene: ../assets/sled_v134.bk3d.gz
	109 meshes, 109 instances
	uninstanced vertices: 348015
	uninstanced triangles: 418036
Minimum samples per face: 3
Generate sample points ... 
  117.97 ms
Total samples: 1254108
Compute AO ...             
	setup ...             335.70 ms
	accum raygen ...        0.17 ms
	accum query ...       793.83 ms
	accum update AO ...     0.35 ms
	copy AO out ...         0.51 ms
 1142.60 ms
Map AO to vertices  ...    
	build mass matrices ...             149.94 ms
	build regularization matrices ...   403.11 ms
	decompose matrices ...              265.34 ms
	solve linear systems ...            10.17 ms
  829.68 ms
~~~




