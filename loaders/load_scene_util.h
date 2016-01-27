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

