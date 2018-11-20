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

#pragma once

#include <cstddef>

// Note: cstdint would require building with c++11 on gcc
#if defined(_WIN32)
#include <cstdint>
#else
#include <stdint.h>
#endif


namespace bake
{


struct SampleInfo
{
  unsigned  tri_idx;
  float     bary[3];
  float     dA;
};


struct Mesh
{
  size_t    num_vertices;
  float*    vertices;
  unsigned  vertex_stride_bytes;
  float*    normals;
  unsigned  normal_stride_bytes;
  size_t    num_triangles;
  unsigned int* tri_vertex_indices;
  float     bbox_min[3];
  float     bbox_max[3];
};

struct Instance
{
  float xform[16];  // 4x4 row major
  uint64_t storage_identifier; // for saving the baked results
  unsigned mesh_index;
  float bbox_min[3];
  float bbox_max[3];
};

// Packages up geometry for routines below.
// A scene is a set of instances that index into meshes. Meshes may be shared between instances.
struct Scene
{
  Mesh* meshes;
  size_t num_meshes;
  Instance* instances;
  size_t num_instances;
};


struct AOSamples
{
  size_t        num_samples;
  float*        sample_positions;
  float*        sample_normals;
  float*        sample_face_normals;
  SampleInfo*   sample_infos;
};

enum VertexFilterMode
{
  VERTEX_FILTER_AREA_BASED=0,
  VERTEX_FILTER_LEAST_SQUARES,
  VERTEX_FILTER_INVALID
};


size_t distributeSamples(
    const Scene&    scene,
    const size_t    min_samples_per_triangle,
    const size_t    requested_num_samples,
    size_t*         num_samples_per_instance // output
    );


void sampleInstances(
    const Scene&  scene,
    const size_t* num_samples_per_instance,
    const size_t  min_samples_per_triangle,
    AOSamples&    ao_samples
    );


void computeAO( 
    const Scene&     scene,
    const AOSamples& ao_samples,
    const int        rays_per_sample,
    const float      scene_offset_scale,
    const float      scene_maxdistance_scale,
    const float*     bbox_min,
    const float*     bbox_max,
    float*           ao_values 
    );


// This version takes extra "blocker" objects that occlude rays,
// but do not have any AO samples of their own.
void computeAOWithBlockers(
    const Scene&     scene,
    const Scene&     blockers,
    const AOSamples& ao_samples,
    const int        rays_per_sample,
    const float      scene_offset_scale,
    const float      scene_maxdistance_scale,
    const float*     bbox_min,
    const float*     bbox_max,
    float*           ao_values 
    );

void mapAOToVertices(
    const Scene&            scene,
    const size_t*           num_samples_per_instance,
    const AOSamples&        ao_samples,
    const float*            ao_values,
    const VertexFilterMode  mode,
    const float             regularization_weight,
    float**                 vertex_ao
    );

void mapAOToTextures(
    );


}
