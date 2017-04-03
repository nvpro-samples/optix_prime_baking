
optix_prime_baking
==================

![Rocket Sled](https://github.com/nvpro-samples/optix_prime_baking/blob/master/doc/sled_multiple_meshes.png)

This sample shows how to precompute ambient occlusion with OptiX Prime, and store it on the
vertices of a mesh for use during final shading with OpenGL.  Also see the accompanying 
[Gameworks blog article](https://developer.nvidia.com/optix-prime-baking-sample).

#### Requirements
  * A recent version of Visual Studio (tested with VS 2013 on Windows 7) or gcc (tested with gcc 4.8.4 on Ubuntu 14.04) 
  * CUDA 7.5+ and matching driver with supported GPU.
  * A recent version of CMake (tested with 2.8.12).
  * OptiX 3.9.0 from the shared_optix git repo (see below).  No separate install of OptiX is needed.
  * On Linux, [GLFW](http://www.glfw.org/) (tested with 3.2.0).

#### How to Build & Run

Quick build instructions:

1) Clone the following nvpro-samples repositories:
  - //github.com/nvpro-samples/build_all.git
  - //github.com/nvpro-samples/shared_optix.git
  - //github.com/nvpro-samples/shared_sources.git
  - //github.com/nvpro-samples/shared_external.git
  - //github.com/nvpro-samples/optix_prime_baking.git

2) Download and install a recent version of [CMake](https://cmake.org)

3) Linux only: download and install [GLFW](http://www.glfw.org/).

4) Open CMake-gui (Windows) or ccmake (Linux):
  - Source code to: /nvpro_ samples/build_all
  - Build folder: /nvpro_samples/build_all/build
  - Optional: toggle MODELS_DOWNLOAD_DISABLED to download extra scenes during the configure step.
  - Configure and select a compiler if prompted.
  - Generate

5) Open the nvpro_samples.sln into Visual Studio, and Build All.  On Linux, 'make' in the build directory.

6) Select the optix_prime_baking sample as the Startup project in VS.

7) Click Run in VS, or run the 'nvpro_samples/bin_x64/optix_prime_baking' binary in Linux.

The sample is configured on the command line; use the "-h" flag to list options or check main.cpp.  The options at the time the sample was created are shown below:

    App options:
      -h  | --help                          Print this usage message
      -f  | --file <scene_file>             Specify model to be rendered (obj, bk3d, bk3d.gz, csf, csf.gz).
      -o  | --outfile <vertex_ao_file>      Specify raw file where per-instance ao vertices are stored (very basic fileformat).
      -i  | --instances <n>                 Number of instances per mesh (default 1).  For testing.
      -r  | --rays    <n>                   Number of rays per sample point for gather (default 64)
      -s  | --samples <n>                   Number of sample points on mesh (default 3 per face; any extra samples are based on area)
      -t  | --samples_per_face <n>          Minimum number of samples per face (default 3)
      -d  | --ray_distance_scale <s>        Distance offset scale for ray from face: ray offset = maximum scene extent * s. (default 0.01)
            --ray_distance <s>              Distance offset scale for ray from face: ray offset = s. (overrides scale-based version, used if non zero)
      -m  | --hit_distance_scale <s>        Maximum hit distance to contribute: max distance = maximum scene extent * s. (default 10)
            --hit_distance <s>              Maximum hit distance to contribute: max distance = s. (overrides scale-based version, used if non zero)
      -g  | --ground_setup <axis> <s> <o>   Ground plane setup: axis(int 0,1,2,3,4,5 = +x,+y,+z,-x,-y,-z) scale(float) offset(float).  (default 1 100 0.03)
            --no_ground_plane               Disable virtual ground plane
      -w  | --regularization_weight <w>     Regularization weight for least squares, positive range. (default 0.1)
            --no_least_squares              Disable least squares filtering
            --no_viewer                     Disable OpenGL viewer
            --no_gpu                        Disable GPU usage in raytracer
            --conserve_memory               Triggers some internal settings in optix to save memory
            
    Viewer keys:
       e                                    Draw mesh edges on/off
       f                                    Frame scene
       q                                    Quit
 
#### Supported scene formats 

Loaders are provided for OBJ, [Bak3d](https://github.com/tlorach/Bak3d) and CSF (basic cad scene file format used in various nvpro-samples).  The OBJ loader flattens all groups into a single mesh.  The bk3d/csf loaders preserve separate meshes, as shown in the teaser image above of a rocket sled with 109 meshes.  Use the utilities in the Bak3d repo to convert other formats, or write a new loader for your favorite format and add it to the "loaders" subdirectory.

The rocket sled .bk3d file is automatically downloaded depending on MODELS_DOWNLOAD_DISABLED in the cmake config.

#### Output format

The sample uses a very basic format, that allows storage of the per-instance vertex-AO values in a binary file.

~~~ cpp
{
  uint64_t num_instances;
  uint64_t num_vertices;
  
  struct Instance {
    uint64_t storage_identifier;  // set by loader
    uint64_t offset_vertices;     // at which index to start in vertex array
    uint64_t num_vertices;        // how many ao vertex values used by instance
  } instances[num_instances];
  
  float ao_values[num_vertices];  // one value per vertex
}
~~~

#### Support

For general OptiX help, please join the NVIDIA Developer Program and download the full [OptiX SDK](https://developer.nvidia.com/optix), then post on the OptiX forums or mailing list.

