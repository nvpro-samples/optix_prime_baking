
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
    const float scene_scale,
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
  const float3 ray_origin       = sample_pos + 0.01f * scene_scale * sample_norm;
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
void bake::generateRaysDevice(unsigned int seed, int px, int py, int sqrt_passes, float scene_scale, const bake::AOSamples& ao_samples, Ray* rays )
{
  const int block_size  = 512;                                                           
  const int block_count = idivCeil( (int)ao_samples.num_samples, block_size );                              

  generateRaysKernel<<<block_count,block_size>>>( 
      seed,
      px,
      py,
      sqrt_passes,
      scene_scale,
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
void updateAOKernel( int num_samples, const float* hit_data, float* ao_data )
{
  int idx = threadIdx.x + blockIdx.x*blockDim.x;                                 
  if( idx >= num_samples )                                                             
    return;

  ao_data[idx] += hit_data[idx] > 0 ? 1.0f : 0.0f;
}

// Precondition: ao output initialized to 0 before first pass
__host__
void bake::updateAODevice( int num_samples, const float* hits, float* ao )
{
  int block_size  = 512;                                                           
  int block_count = idivCeil( num_samples, block_size );                              

  updateAOKernel<<<block_count,block_size>>>( num_samples, hits, ao );
}


