/* Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


#include "bake_kernels.h"
#include "bake_api.h"
#include "random.h"

#include <optixu/optixu_math_namespace.h>
using optix::float3;



inline int idivCeil( int x, int y )                                              
{                                                                                
    return (x + y-1)/y;                                                            
}


//------------------------------------------------------------------------------
//
// Ray generation kernel
// 
//------------------------------------------------------------------------------
__global__
void generateRaysKernel( 
    const unsigned int base_seed,
    const int px,
    const int py,
    const int sqrt_passes,
    const float scene_offset,
    const int num_samples,
    const float3* sample_normals,
    const float3* sample_face_normals,
    const float3* sample_positions,
    Ray* rays
    )
{
  int idx = threadIdx.x + blockIdx.x*blockDim.x;                                 
  if( idx >= num_samples )                                                             
    return;

  const unsigned int tea_seed = (base_seed << 16) | (px*sqrt_passes+py);
  unsigned seed = tea<2>( tea_seed, idx );

  const float3 sample_norm      = sample_normals[idx]; 
  const float3 sample_face_norm = sample_face_normals[idx];
  const float3 sample_pos       = sample_positions[idx];
  const float3 ray_origin       = sample_pos + scene_offset * sample_norm;
  optix::Onb onb( sample_norm );

  float3 ray_dir;
  float u0 = ( static_cast<float>( px ) + rnd( seed ) ) / static_cast<float>( sqrt_passes );
  float u1 = ( static_cast<float>( py ) + rnd( seed ) ) / static_cast<float>( sqrt_passes );
  int j = 0;
  do
  {
    optix::cosine_sample_hemisphere( u0, u1, ray_dir );

    onb.inverse_transform( ray_dir );
    ++j;
    u0 = rnd( seed );
    u1 = rnd( seed );
  }
  while( j < 5 && optix::dot( ray_dir, sample_face_norm ) <= 0.0f );
    
  rays[idx].origin    = ray_origin; 
  rays[idx].direction = ray_dir;

}

__host__
void bake::generateRaysDevice(unsigned int seed, int px, int py, int sqrt_passes, float scene_offset, const bake::AOSamples& ao_samples, Ray* rays )
{
  const int block_size  = 512;                                                           
  const int block_count = idivCeil( (int)ao_samples.num_samples, block_size );                              

  generateRaysKernel<<<block_count,block_size>>>( 
      seed,
      px,
      py,
      sqrt_passes,
      scene_offset,
      (int)ao_samples.num_samples,
      (float3*)ao_samples.sample_normals,
      (float3*)ao_samples.sample_face_normals,
      (float3*)ao_samples.sample_positions,
      rays
      );
}

//------------------------------------------------------------------------------
//
// AO update kernel
// 
//------------------------------------------------------------------------------

__global__
void updateAOKernel(int num_samples, float maxdistance, const float* hit_data, float* ao_data)
{
  int idx = threadIdx.x + blockIdx.x*blockDim.x;                                 
  if( idx >= num_samples )                                                             
    return;

  float distance = hit_data[idx];
  ao_data[idx] += distance > 0.0 && distance < maxdistance ? 1.0f : 0.0f;
}

// Precondition: ao output initialized to 0 before first pass
__host__
void bake::updateAODevice( int num_samples, float maxdistance, const float* hits, float* ao )
{
  int block_size  = 512;                                                           
  int block_count = idivCeil( num_samples, block_size );                              

  updateAOKernel <<<block_count, block_size >>>(num_samples, maxdistance, hits, ao);
}


