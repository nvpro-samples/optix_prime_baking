
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
#include <optixu/optixu_matrix_namespace.h>

#include "random.h"

#include <algorithm>
#include <float.h>
#include <iostream>
#include <map>

using namespace optix::prime;

#define ACCUM_TIME( t, x )        \
do {                              \
  t.start();                      \
  x;                              \
  t.stop();                       \
} while( false )

namespace
{

void createInstances( Context& context, const bake::Instance* instances, const size_t num_instances, 
    std::vector<Model>& models, std::vector<RTPmodel>& prime_instances, std::vector<optix::Matrix4x4>& transforms )
{
  std::map<bake::Mesh*, RTPmodel> unique_meshes;

  for (int i = 0; i < num_instances; ++i) {
    RTPmodel rtp_model(0);
    // Share mesh between instances if possible
    if (unique_meshes.find(instances[i].mesh) != unique_meshes.end()) {
      rtp_model = unique_meshes.find(instances[i].mesh)->second;
    } else {
      // Allocate new model
      Model model = context->createModel();
      const bake::Mesh& mesh = *instances[i].mesh;
      model->setTriangles(
          mesh.num_triangles, RTP_BUFFER_TYPE_HOST, mesh.tri_vertex_indices,
          mesh.num_vertices,  RTP_BUFFER_TYPE_HOST, mesh.vertices
          );
      model->update( 0 );
      models.push_back(model);  // Model is ref counted, so need to return it to prevent destruction
      rtp_model = model->getRTPmodel();
      unique_meshes.insert(std::make_pair(instances[i].mesh, rtp_model));
    }

    prime_instances.push_back(rtp_model);
    transforms.push_back(optix::Matrix4x4(instances[i].xform));
  }

}

inline size_t idivCeil( size_t x, size_t y )                                              
{                                                                                
    return (x + y-1)/y;                                                            
}


} // end namespace


void bake::ao_optix_prime(
    const bake::Instance* instances,
    const size_t num_instances,
    const bake::Instance* blockers,
    const size_t num_blockers,
    const bake::AOSamples* ao_samples_per_instance,
    const int rays_per_sample,
    float** ao_values
    )
{

  Timer setup_timer;
  setup_timer.start( );

  Context ctx = Context::create( RTP_CONTEXT_TYPE_CUDA );

  std::vector<Model> models;
  std::vector<RTPmodel> prime_instances;
  std::vector<optix::Matrix4x4> transforms;
  createInstances( ctx, instances, num_instances, models, prime_instances, transforms );
  if (num_blockers > 0) {
    createInstances( ctx, blockers, num_blockers, models, prime_instances, transforms ); 
  }
  Model scene = ctx->createModel();
  scene->setInstances( prime_instances.size(), RTP_BUFFER_TYPE_HOST, &prime_instances[0],
                      RTP_BUFFER_FORMAT_TRANSFORM_FLOAT4x4, RTP_BUFFER_TYPE_HOST, &transforms[0] );
  scene->update( 0 );

  Query   query = scene->createQuery( RTP_QUERY_TYPE_ANY );

  const int sqrt_rays_per_sample = static_cast<int>( sqrtf( static_cast<float>( rays_per_sample ) ) + .5f );

  Timer raygen_timer;
  Timer query_timer;
  Timer updateao_timer;
  Timer copyao_timer;

  for (size_t idx = 0; idx < num_instances; ++idx) {

    // Split sample points into batches
    const size_t batch_size = 2000000;  // Note: fits in GTX 750
    const bake::AOSamples& ao_samples = ao_samples_per_instance[idx];
    const size_t num_batches = std::max(idivCeil(ao_samples.num_samples, batch_size), size_t(1));

    for (size_t batch_idx = 0; batch_idx < num_batches; batch_idx++) {

      const size_t sample_offset = batch_idx*batch_size;
      const size_t num_samples = std::min(batch_size, ao_samples.num_samples - sample_offset);

      // Copy all necessary data to device
      Buffer<float3> sample_normals     ( num_samples, RTP_BUFFER_TYPE_CUDA_LINEAR );
      Buffer<float3> sample_face_normals( num_samples, RTP_BUFFER_TYPE_CUDA_LINEAR );
      Buffer<float3> sample_positions   ( num_samples, RTP_BUFFER_TYPE_CUDA_LINEAR );
      
      cudaMemcpy( sample_normals.ptr(),      ao_samples.sample_normals + 3*sample_offset,      sample_normals.sizeInBytes(),  cudaMemcpyHostToDevice );
      cudaMemcpy( sample_face_normals.ptr(), ao_samples.sample_face_normals + 3*sample_offset, sample_face_normals.sizeInBytes(),  cudaMemcpyHostToDevice );
      cudaMemcpy( sample_positions.ptr(),    ao_samples.sample_positions + 3*sample_offset,    sample_positions.sizeInBytes(),  cudaMemcpyHostToDevice );
      bake::AOSamples ao_samples_device;
      ao_samples_device.num_samples = num_samples;
      ao_samples_device.sample_normals      = reinterpret_cast<float*>( sample_normals.ptr() );
      ao_samples_device.sample_face_normals = reinterpret_cast<float*>( sample_face_normals.ptr() );
      ao_samples_device.sample_positions    = reinterpret_cast<float*>( sample_positions.ptr() );
      ao_samples_device.sample_infos = 0;

      Buffer<float> hits( num_samples, RTP_BUFFER_TYPE_CUDA_LINEAR );
      Buffer<Ray>   rays( num_samples, RTP_BUFFER_TYPE_CUDA_LINEAR );
      Buffer<float> ao  ( num_samples, RTP_BUFFER_TYPE_CUDA_LINEAR );
      cudaMemset( ao.ptr(), 0, ao.sizeInBytes() );
      
      query->setRays( rays.count(), Ray::format,             rays.type(), rays.ptr() );
      query->setHits( hits.count(), RTP_BUFFER_FORMAT_HIT_T, hits.type(), hits.ptr() );
      setup_timer.stop();

      const optix::Matrix4x4 xform(instances[idx].xform);
      const float* bmin = instances[idx].mesh->bbox_min;
      const float* bmax = instances[idx].mesh->bbox_max;
      const float3 bbox_min = make_float3(xform*make_float4(bmin[0], bmin[1], bmin[2], 1.0f));
      const float3 bbox_max = make_float3(xform*make_float4(bmax[0], bmax[1], bmax[2], 1.0f));
      const float scene_scale = std::max( std::max(bbox_max.x - bbox_min.x,
                                                   bbox_max.y - bbox_min.y),
                                                   bbox_max.z - bbox_min.z );

      const unsigned int seed = idx;
      for( int i = 0; i < sqrt_rays_per_sample; ++i )
      for( int j = 0; j < sqrt_rays_per_sample; ++j )
      {
        ACCUM_TIME( raygen_timer,   generateRaysDevice(seed, i, j, sqrt_rays_per_sample, scene_scale, ao_samples_device, rays.ptr() ) );
        ACCUM_TIME( query_timer,    query->execute( 0 ) );
        ACCUM_TIME( updateao_timer, updateAODevice( (int)num_samples, hits.ptr(), ao.ptr() ) );
      }

      // copy ao to ao_values
      copyao_timer.start();
      cudaMemcpy( &ao_values[idx][sample_offset], ao.ptr(), ao.sizeInBytes(), cudaMemcpyDeviceToHost ); 
      copyao_timer.stop();
    }

    // normalize
    for( size_t  i = 0; i < ao_samples.num_samples; ++i ) {
      ao_values[idx][i] = 1.0f - ao_values[idx][i] / rays_per_sample; 
    }

  }

  std::cerr << "\n\tsetup ...           ";  printTimeElapsed( setup_timer );
  std::cerr << "\taccum raygen ...    ";  printTimeElapsed( raygen_timer );
  std::cerr << "\taccum query ...     ";  printTimeElapsed( query_timer );
  std::cerr << "\taccum update AO ... ";  printTimeElapsed( updateao_timer );
  std::cerr << "\tcopy AO out ...     ";  printTimeElapsed( copyao_timer );
}



