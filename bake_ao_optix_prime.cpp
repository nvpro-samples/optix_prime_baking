/*-----------------------------------------------------------------------
  Copyright (c) 2015-2016, NVIDIA. All rights reserved.
  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions
  are met:
   * Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.
   * Neither the name of its contributors may be used to endorse 
     or promote products derived from this software without specific
     prior written permission.
  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
  PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
  OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
-----------------------------------------------------------------------*/

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

#define ACCUM_TIME( t, x )        \
do {                              \
  t.start();                      \
  x;                              \
  t.stop();                       \
} while( false )

namespace
{


// For scoping Prime data that needs to stay around during queries
struct PrimeSceneData {

  std::vector<Buffer<float3>* > vertex_buffers;  // pointers since Buffers can't be copied
  std::vector<Buffer<int3>* >   index_buffers;
  std::vector<optix::prime::Model> models;

  virtual ~PrimeSceneData() {
    // clean up Buffer pointers.
    for (size_t i = 0; i < vertex_buffers.size(); ++i) {
      delete vertex_buffers[i];
    }
    vertex_buffers.clear();
    for (size_t i = 0; i < index_buffers.size(); ++i) {
      delete index_buffers[i];
    }
    index_buffers.clear();
  }
};


// Build and return a two-level Prime scene that is ready for ray queries.

optix::prime::Model createPrimeScene( optix::prime::Context& context, const RTPcontexttype context_type,
    const bake::Mesh* meshes, const size_t num_meshes, const bake::Instance* instances, const size_t num_instances, 
    const bool conserve_memory,
    // output
    PrimeSceneData& psd )
{

  // Create one Prime model per input mesh.  Each of these models will have its own accel structure; this is the lower
  // level of a two-level scene hierarchy.

  psd.models.reserve( num_meshes );
  for (size_t meshIdx = 0; meshIdx < num_meshes; ++meshIdx) {
    optix::prime::Model model = context->createModel();
    if (conserve_memory){
      model->setBuilderParameter(RTP_BUILDER_PARAM_USE_CALLER_TRIANGLES, 1);
      model->setBuilderParameter<size_t>(RTP_BUILDER_PARAM_CHUNK_SIZE, 512 * 1024 * 1024);
    }
    psd.models.push_back(model);

    // Delay building accels until we connect buffers below
  }

  if ( context_type == RTP_CONTEXT_TYPE_CUDA ) {

    // Create device buffers for vertices and indices.  We could let Prime handle this copy, 
    // but doing it ourselves preserves any sharing of buffers between meshes, e.g. from bak3d file format.

    std::map< float*, Buffer<float3>* > unique_vertex_buffers;
    std::map< unsigned int*, Buffer<int3>* > unique_index_buffers;

    for (size_t meshIdx = 0; meshIdx < num_meshes; ++meshIdx) {
      
      const bake::Mesh& mesh = meshes[meshIdx];

      // Verts
      Buffer<float3>* vertex_buffer = NULL;
      if (unique_vertex_buffers.find(mesh.vertices) != unique_vertex_buffers.end()) {
        vertex_buffer = unique_vertex_buffers.find(mesh.vertices)->second;
      } else {
        vertex_buffer = new Buffer<float3>( mesh.num_vertices, RTP_BUFFER_TYPE_CUDA_LINEAR, UNLOCKED, mesh.vertex_stride_bytes );
        cudaMemcpy( vertex_buffer->ptr(), mesh.vertices, vertex_buffer->sizeInBytes(), cudaMemcpyHostToDevice );
        unique_vertex_buffers[mesh.vertices] = vertex_buffer;

        // Don't leak the buffer
        psd.vertex_buffers.push_back(vertex_buffer);
      }
      
      // Indices
      Buffer<int3>* index_buffer = NULL;
      if (unique_index_buffers.find(mesh.tri_vertex_indices) != unique_index_buffers.end()) {
        index_buffer = unique_index_buffers.find(mesh.tri_vertex_indices)->second;
      } else {
        index_buffer = new Buffer<int3>( mesh.num_triangles, RTP_BUFFER_TYPE_CUDA_LINEAR );
        cudaMemcpy( index_buffer->ptr(), mesh.tri_vertex_indices, index_buffer->sizeInBytes(), cudaMemcpyHostToDevice );
        unique_index_buffers[mesh.tri_vertex_indices] = index_buffer;
        psd.index_buffers.push_back(index_buffer);
      }

      // Connect device buffers to model
      psd.models[meshIdx]->setTriangles(
          index_buffer->count(), index_buffer->type(), index_buffer->ptr(),
          vertex_buffer->count(), vertex_buffer->type(), vertex_buffer->ptr(), vertex_buffer->stride()
          );

      // Build the accel on the device.  This is a blocking call.
      psd.models[meshIdx]->update( 0 );
    }

  } else {
    // CPU context: just hand Prime the pointers we already have, no need for another copy
    for (size_t meshIdx = 0; meshIdx < num_meshes; ++meshIdx) {

      const bake::Mesh& mesh = meshes[meshIdx];

      // Connect host buffers to model
      psd.models[meshIdx]->setTriangles(
          mesh.num_triangles, RTP_BUFFER_TYPE_HOST, mesh.tri_vertex_indices,
          mesh.num_vertices,  RTP_BUFFER_TYPE_HOST, mesh.vertices, mesh.vertex_stride_bytes
          );

      // Build the accel on the host.
      psd.models[meshIdx]->update( 0 );

    }
  }

  // The lower level of the scene hierarchy is done, now build the upper level of instances.

  std::vector<RTPmodel> rtp_models;
  std::vector<optix::Matrix4x4> transforms;
  rtp_models.reserve(num_instances);
  transforms.reserve(num_instances);
  for (int i = 0; i < num_instances; ++i) {
    const size_t index = instances[i].mesh_index;
    RTPmodel rtp_model = psd.models[index]->getRTPmodel();
    rtp_models.push_back(rtp_model);
    transforms.push_back(optix::Matrix4x4(instances[i].xform));
  }

  optix::prime::Model scene_model = context->createModel();
  scene_model->setInstances( rtp_models.size(), RTP_BUFFER_TYPE_HOST, &rtp_models[0],
                      RTP_BUFFER_FORMAT_TRANSFORM_FLOAT4x4, RTP_BUFFER_TYPE_HOST, &transforms[0] );
  scene_model->update( 0 );

  return scene_model;
}


inline size_t idivCeil( size_t x, size_t y )                                              
{                                                                                
    return (x + y-1)/y;                                                            
}


} // end namespace


void bake::ao_optix_prime(
    const Scene& scene,
    const bake::AOSamples& ao_samples,
    const int rays_per_sample,
    const float  scene_offset,
    const float  scene_maxdistance,
    const bool   cpu_mode,
    const bool   conserve_memory,
    float* ao_values
    )
{

  Timer setup_timer;
  setup_timer.start( );

  const RTPcontexttype context_type = cpu_mode ? RTP_CONTEXT_TYPE_CPU : RTP_CONTEXT_TYPE_CUDA;
  optix::prime::Context ctx = optix::prime::Context::create( context_type );

  PrimeSceneData psd;

  optix::prime::Model prime_scene = createPrimeScene( ctx, context_type, scene.meshes, scene.num_meshes, scene.instances, scene.num_instances, conserve_memory, psd );

  optix::prime::Query query = prime_scene->createQuery( RTP_QUERY_TYPE_ANY );

  const int sqrt_rays_per_sample = static_cast<int>( sqrtf( static_cast<float>( rays_per_sample ) ) + .5f );
  setup_timer.stop();

  Timer raygen_timer;
  Timer query_timer;
  Timer updateao_timer;
  Timer copyao_timer;

  unsigned seed = 0;

  // Split sample points into batches to help limit device memory usage.
  const size_t batch_size = 2000000;  // Note: fits on GTX 750 (1 GB) along with Hunter model
  const size_t num_batches = std::max(idivCeil(ao_samples.num_samples, batch_size), size_t(1));

  for (size_t batch_idx = 0; batch_idx < num_batches; batch_idx++, seed++) {

    setup_timer.start();
    const size_t sample_offset = batch_idx*batch_size;
    const size_t num_samples = std::min(batch_size, ao_samples.num_samples - sample_offset);

    // Copy sample points to device
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

    for( int i = 0; i < sqrt_rays_per_sample; ++i )
    for( int j = 0; j < sqrt_rays_per_sample; ++j )
    {
      ACCUM_TIME(raygen_timer,    generateRaysDevice(seed, i, j, sqrt_rays_per_sample, scene_offset, scene_maxdistance, ao_samples_device, rays.ptr()));

      // Host or device, depending on Prime context type.  For a host context, which we assume is rare, Prime will copy the rays from device to host.
      ACCUM_TIME( query_timer,    query->execute( 0 ) );

      ACCUM_TIME(updateao_timer,  updateAODevice((int)num_samples, hits.ptr(), ao.ptr()));
    }

    ACCUM_TIME(updateao_timer, normalizeAODevice((int)num_samples, ao.ptr(), rays_per_sample));

    // copy AO values back to host
    copyao_timer.start();
    cudaMemcpy( &ao_values[sample_offset], ao.ptr(), ao.sizeInBytes(), cudaMemcpyDeviceToHost ); 
    copyao_timer.stop();
  }
  
  std::cerr << "\n\tsetup ...           ";  printTimeElapsed( setup_timer );
  std::cerr << "\taccum raygen ...    ";  printTimeElapsed( raygen_timer );
  std::cerr << "\taccum query ...     ";  printTimeElapsed( query_timer );
  std::cerr << "\taccum update AO ... ";  printTimeElapsed( updateao_timer );
  std::cerr << "\tcopy AO out ...     ";  printTimeElapsed( copyao_timer );
}



