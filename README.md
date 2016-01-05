
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


The benefit of least squares filtering is shown below on the default teapot.obj mesh.

TODO

There are command line arguments for the input mesh, number of sample points, number of rays, etc.  Use the "-h" flag or
check main.cpp.
 
