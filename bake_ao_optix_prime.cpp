
//
// Copyright (c) 2015 NVIDIA Corporation.  All rights reserved.
// 
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto.  Any use, reproduction, disclosure or distribution of
// this software and related documentation without an express license agreement
// from NVIDIA Corporation is strictly prohibited.
// 
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF
// MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL
// NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR
// CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR
// LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF BUSINESS
// INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
// INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGES
//

#include "bake_ao_optix_prime.h"
#include "bake_kernels.h"
#include "bake_util.h"

#include "Buffer.h"
#include <optix_prime/optix_primepp.h>
#include <optixu/optixu_math_namespace.h>

#include "random.h"

#include <algorithm>
#include <iostream>

using namespace optix::prime;

#define ACCUM_TIME( t, x )        \
do {                              \
  t.start();                      \
  x;                              \
  t.stop();                       \
} while( false )

namespace
{

Model createModel( Context& context, const bake::Mesh& mesh )
{
  Model model = context->createModel();
  model->setTriangles(
      mesh.num_triangles, RTP_BUFFER_TYPE_HOST, mesh.tri_vertex_indices,
      mesh.num_vertices,  RTP_BUFFER_TYPE_HOST, mesh.vertices
      );
  model->update( 0 );
  return model;
}


} // end namespace


void bake::ao_optix_prime(
    const bake::Mesh& mesh,
    const bake::AOSamples& ao_samples,
    const int rays_per_sample,
    float* ao_values
    )
{

  Context ctx = Context::create( RTP_CONTEXT_TYPE_CUDA );
  Model   model = createModel( ctx, mesh );
  Query   query = model->createQuery( RTP_QUERY_TYPE_ANY );

  Timer setup_timer;
  setup_timer.start();

  // Copy all necessary data to device
  Buffer<float3> sample_normals     ( ao_samples.num_samples, RTP_BUFFER_TYPE_CUDA_LINEAR );
  Buffer<float3> sample_face_normals( ao_samples.num_samples, RTP_BUFFER_TYPE_CUDA_LINEAR );
  Buffer<float3> sample_positions   ( ao_samples.num_samples, RTP_BUFFER_TYPE_CUDA_LINEAR );
  
  cudaMemcpy( sample_normals.ptr(),      ao_samples.sample_normals,      sample_normals.sizeInBytes(),      cudaMemcpyHostToDevice );
  cudaMemcpy( sample_face_normals.ptr(), ao_samples.sample_face_normals, sample_face_normals.sizeInBytes(), cudaMemcpyHostToDevice );
  cudaMemcpy( sample_positions.ptr(),    ao_samples.sample_positions,    sample_positions.sizeInBytes(),    cudaMemcpyHostToDevice );
  bake::AOSamples ao_samples_device;
  ao_samples_device.num_samples = ao_samples.num_samples;
  ao_samples_device.sample_normals      = reinterpret_cast<float*>( sample_normals.ptr() );
  ao_samples_device.sample_face_normals = reinterpret_cast<float*>( sample_face_normals.ptr() );
  ao_samples_device.sample_positions    = reinterpret_cast<float*>( sample_positions.ptr() );
  ao_samples_device.sample_infos = 0;

  Buffer<float> hits( ao_samples.num_samples, RTP_BUFFER_TYPE_CUDA_LINEAR );
  Buffer<Ray>   rays( ao_samples.num_samples, RTP_BUFFER_TYPE_CUDA_LINEAR );
  Buffer<float> ao  ( ao_samples.num_samples, RTP_BUFFER_TYPE_CUDA_LINEAR );
  cudaMemset( ao.ptr(), 0, ao.sizeInBytes() );
  
  query->setRays( rays.count(), Ray::format,             rays.type(), rays.ptr() );
  query->setHits( hits.count(), RTP_BUFFER_FORMAT_HIT_T, hits.type(), hits.ptr() );
  setup_timer.stop();

  Timer raygen_timer;
  Timer query_timer;
  Timer updateao_timer;

  const int sqrt_rays_per_sample = static_cast<int>( sqrtf( static_cast<float>( rays_per_sample ) ) + .5f );

  const float scene_scale = std::max( std::max(mesh.bbox_max[0] - mesh.bbox_min[0],
                                               mesh.bbox_max[1] - mesh.bbox_min[1]),
                                               mesh.bbox_max[2] - mesh.bbox_min[2] );
  float* frays = reinterpret_cast<float*>( rays.ptr() );
  for( int i = 0; i < sqrt_rays_per_sample; ++i )
  for( int j = 0; j < sqrt_rays_per_sample; ++j )
  {
    ACCUM_TIME( raygen_timer,   generateRaysDevice( i, j, sqrt_rays_per_sample, scene_scale, ao_samples_device, frays ) );
    ACCUM_TIME( query_timer,    query->execute( 0 ) );
    ACCUM_TIME( updateao_timer, updateAODevice( (int)ao_samples.num_samples, hits.ptr(), ao.ptr() ) );
  }

  // copy ao to ao_values
  Timer copyao_timer;
  copyao_timer.start();
  cudaMemcpy( ao_values, ao.ptr(), ao.sizeInBytes(), cudaMemcpyDeviceToHost ); 
  for( size_t  i = 0; i < ao.count(); ++i )
    ao_values[i] = 1.0f - ao_values[i] / rays_per_sample; 
  copyao_timer.stop();

  std::cerr << "\n\tsetup ...           ";  printTimeElapsed( setup_timer );
  std::cerr << "\taccum raygen ...    ";  printTimeElapsed( raygen_timer );
  std::cerr << "\taccum query ...     ";  printTimeElapsed( query_timer );
  std::cerr << "\taccum update AO ... ";  printTimeElapsed( updateao_timer );
  std::cerr << "\tcopy AO out ...     ";  printTimeElapsed( copyao_timer );
}



