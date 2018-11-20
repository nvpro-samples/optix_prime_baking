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

#include "bake_filter.h"
#include "bake_api.h"

#include <cassert>
#include <iostream>
#include <vector>

#include <optixu/optixu_math_namespace.h>


using namespace optix;

namespace {

// Splat area-weighted samples onto vertices
void filter_mesh_area_weighted(
    const bake::Mesh&       mesh,
    const bake::AOSamples&  ao_samples,
    const float*            ao_values,
    float*                  vertex_ao
    )
{
  std::vector<double> weights(mesh.num_vertices, 0.0);
  std::fill(vertex_ao, vertex_ao + mesh.num_vertices, 0.0f);

  const int3* tri_vertex_indices  = reinterpret_cast<int3*>( mesh.tri_vertex_indices );

  for (size_t i = 0; i < ao_samples.num_samples; ++i) {
    const bake::SampleInfo& info = ao_samples.sample_infos[i];
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

}  // namespace

void bake::filter(
    const Scene&         scene,
    const size_t*        num_samples_per_instance,
    const AOSamples&     ao_samples,
    const float*         ao_values,
    float**              vertex_ao
    )
{
  std::vector<size_t> sample_offset_per_instance(scene.num_instances);
  {
    size_t sample_offset = 0;
    for (size_t i = 0; i < scene.num_instances; ++i) {
      sample_offset_per_instance[i] = sample_offset;
      sample_offset += num_samples_per_instance[i];
    }
  }

#pragma omp parallel for
  for (ptrdiff_t i = 0; i < ptrdiff_t(scene.num_instances); ++i) {
    size_t sample_offset = sample_offset_per_instance[i];

    // Point to samples for this instance
    AOSamples instance_ao_samples;
    instance_ao_samples.num_samples = num_samples_per_instance[i];
    instance_ao_samples.sample_positions = ao_samples.sample_positions + 3*sample_offset;
    instance_ao_samples.sample_normals = ao_samples.sample_normals + 3*sample_offset;
    instance_ao_samples.sample_face_normals = ao_samples.sample_face_normals + 3*sample_offset;
    instance_ao_samples.sample_infos = ao_samples.sample_infos + sample_offset;

    const float* instance_ao_values = ao_values + sample_offset;

    filter_mesh_area_weighted(scene.meshes[scene.instances[i].mesh_index], instance_ao_samples, instance_ao_values, vertex_ao[i]);
  }
}


