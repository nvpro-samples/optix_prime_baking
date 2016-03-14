
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

#pragma once

#include <cstddef>
#include <cstdint>


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
