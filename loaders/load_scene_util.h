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

#pragma once

#include <vector_types.h>
#include <optixu/optixu_matrix_namespace.h>

#include <algorithm>
#include <vector>

inline void xform_bbox(const optix::Matrix4x4& mat, const float in_min[3], const float in_max[3],
                float out_min[3], float out_max[3])
{
  float4 a = mat*optix::make_float4( in_min[0], in_min[1], in_min[2], 1.0f);
  float4 b = mat*optix::make_float4( in_max[0], in_max[1], in_max[2], 1.0f);
  for (size_t k = 0; k < 3; ++k) {
    out_min[k] = (&a.x)[k];
    out_max[k] = (&b.x)[k];
  }
}

inline void expand_bbox(float bbox_min[3], float bbox_max[3], float v[3])
{
  for (size_t k = 0; k < 3; ++k) {
    bbox_min[k] = std::min(bbox_min[k], v[k]);
    bbox_max[k] = std::max(bbox_max[k], v[k]);
  }
}

namespace bake {
  struct Mesh;
  struct Instance;
}

void make_debug_instances(std::vector<bake::Mesh>& meshes, std::vector<bake::Instance>& instances, size_t n, float scene_bbox_min[3], float scene_bbox_max[3]);

