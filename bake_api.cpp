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

#include "bake_api.h"
#include "bake_ao_optix_prime.h"
#include "bake_filter.h"
#include "bake_filter_least_squares.h"
#include "bake_sample.h"
#include <optixu/optixu_math_namespace.h>
#include <cassert>


using namespace optix;


void bake::computeAO( 
    const Scene&      scene,
    const AOSamples&  ao_samples,
    const int         rays_per_sample,
    const float       scene_offset_scale,
    const float       scene_maxdistance_scale,
    const float*      bbox_min,
    const float*      bbox_max,
    float*            ao_values 
    )
{
  Scene blockers = {0};
  bake::ao_optix_prime( scene, blockers,
    ao_samples, rays_per_sample, scene_offset_scale, scene_maxdistance_scale, bbox_min, bbox_max, ao_values);

}

void bake::computeAOWithBlockers(
    const Scene&      scene,
    const Scene&      blockers,
    const AOSamples&  ao_samples,
    const int         rays_per_sample,
    const float       scene_offset_scale,
    const float       scene_maxdistance_scale,
    const float*      bbox_min,
    const float*      bbox_max,
    float*            ao_values 
    )
{

  bake::ao_optix_prime( scene, blockers,
    ao_samples, rays_per_sample, scene_offset_scale, scene_maxdistance_scale, bbox_min, bbox_max, ao_values);

}


size_t bake::distributeSamples(
    const Scene&    scene,
    const size_t    min_samples_per_triangle,
    const size_t    requested_num_samples,
    size_t*         num_samples_per_instance
    )
{

  return bake::distribute_samples( scene, min_samples_per_triangle, requested_num_samples, num_samples_per_instance );

}


void bake::sampleInstances(
    const Scene&  scene,
    const size_t* num_samples_per_instance,
    const size_t  min_samples_per_triangle,
    AOSamples&    ao_samples
    )
{

  bake::sample_instances( scene, num_samples_per_instance, min_samples_per_triangle, ao_samples );

}

void bake::mapAOToVertices(
    const Scene&            scene,
    const size_t*           num_samples_per_instance,
    const AOSamples&        ao_samples,
    const float*            ao_values,
    const VertexFilterMode  mode,
    const float             regularization_weight,
    float**                 vertex_ao
    )
{
    if (mode == VERTEX_FILTER_AREA_BASED) {
      bake::filter( scene, num_samples_per_instance, ao_samples, ao_values, vertex_ao ); 
    } else if (mode == VERTEX_FILTER_LEAST_SQUARES) {
      bake::filter_least_squares( scene, num_samples_per_instance, ao_samples, ao_values, regularization_weight, vertex_ao ); 
    } else {
      assert(0 && "invalid vertex filter mode");
    }
}


void bake::mapAOToTextures(
    )
{
}


