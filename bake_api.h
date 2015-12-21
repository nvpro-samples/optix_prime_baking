
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
  size_t    num_normals;
  float*    normals;
  size_t    num_triangles;
  int*      tri_vertex_indices;
  int*      tri_normal_indices;
  float     bbox_min[3];
  float     bbox_max[3];
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


void sampleSurface(
    const Mesh& mesh,
    size_t min_samples_per_triangle,
    AOSamples&  ao_samples
    );


void computeAO( 
    const Mesh&       mesh,
    const AOSamples&  ao_samples,
    int               rays_per_sample,
    float*            ao_values 
    );


void mapAOToVertices(
    const Mesh&             mesh,
    const AOSamples&        ao_samples,
    const float*            ao_values,
    const VertexFilterMode  mode,
    const float             regularization_weight,
    float*                  vertex_ao
    );


void mapAOToTextures(
    );


}
