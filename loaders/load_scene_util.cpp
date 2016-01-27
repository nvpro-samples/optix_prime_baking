#include "load_scene_util.h"
#include "../bake_api.h"

#include <vector_types.h>
#include <optixu/optixu_matrix_namespace.h>

void make_debug_instances(std::vector<bake::Mesh>& meshes, std::vector<bake::Instance>& instances, size_t n, float scene_bbox_min[3], float scene_bbox_max[3])
{
  // Make up a transform per instance
  const float3 bbox_base = optix::make_float3(0.5f*(scene_bbox_min[0] + scene_bbox_max[0]),
                                                    scene_bbox_min[1],
                                              0.5f*(scene_bbox_min[2] + scene_bbox_max[2]));
  const float rot = 0.5236f;  // pi/6
  const float3 rot_axis = optix::make_float3(0.0f, 1.0f, 0.0f);
  const float scale_factor = 0.9f;
  float scale = scale_factor;
  const float3 base_translation = 1.01*optix::make_float3(scene_bbox_max[0] - scene_bbox_min[0], 0.0f, 0.0f);
  float3 translation = scale_factor* base_translation;

  const size_t original_num_instances = instances.size();
  for (size_t i = 0; i < n; i++) {

   const optix::Matrix4x4 xform = optix::Matrix4x4::translate(translation) *
                                  optix::Matrix4x4::translate(bbox_base) *
                                  optix::Matrix4x4::rotate((i+1)*rot, rot_axis) *
                                  optix::Matrix4x4::scale(optix::make_float3(scale)) *
                                  optix::Matrix4x4::translate(-bbox_base);

    for (size_t idx = 0; idx < original_num_instances; ++idx) {
      bake::Instance instance;
      const unsigned mesh_index = instances[idx].mesh_index;
      instance.mesh_index = mesh_index;

      optix::Matrix4x4 mat = xform*optix::Matrix4x4(instances[idx].xform);
      xform_bbox(mat, meshes[mesh_index].bbox_min, meshes[mesh_index].bbox_max, instance.bbox_min, instance.bbox_max);
      expand_bbox(scene_bbox_min, scene_bbox_max, instance.bbox_min);
      expand_bbox(scene_bbox_min, scene_bbox_max, instance.bbox_max);

      const float* matdata = mat.getData();
      std::copy(matdata, matdata+16, instance.xform);
      instances.push_back(instance);
    }

    scale *= scale_factor;
    translation += scale*base_translation;
  }

}

