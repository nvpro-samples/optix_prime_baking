
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

#include "bake_filter.h"
#include "bake_api.h"

#include <cassert>
#include <iostream>
#include <vector>

#include <optixu/optixu_math_namespace.h>


using namespace optix;


// Splat area-weighted samples onto vertices
void bake::filter(
    const Mesh&       mesh,
    const AOSamples&  ao_samples,
    const float*      ao_values,
    float*            vertex_ao
    )
{
  std::vector<double> weights(mesh.num_vertices, 0.0);
  std::fill(vertex_ao, vertex_ao + mesh.num_vertices, 0.0f);

  const int3* tri_vertex_indices  = reinterpret_cast<int3*>( mesh.tri_vertex_indices );

  for (size_t i = 0; i < ao_samples.num_samples; ++i) {
    const SampleInfo& info = ao_samples.sample_infos[i];
    const int3& tri = tri_vertex_indices[info.tri_idx];

    const float val = ao_values[i];
    const float w[] = {info.dA * info.bary[0],
      info.dA * info.bary[1],
      info.dA * info.bary[2]};

    vertex_ao[tri.x] += w[0] * val;
    weights  [tri.x] += w[0];
    vertex_ao[tri.y] += w[1] * val;
    weights  [tri.y] += w[1];
    vertex_ao[tri.z] += w[2] * val;
    weights  [tri.z] += w[2];
  }

  // Normalize
  for (size_t k = 0; k < mesh.num_vertices; ++k) {
    if (weights[k] > 0.0) vertex_ao[k] /= static_cast<float>(weights[k]);
  }

}

