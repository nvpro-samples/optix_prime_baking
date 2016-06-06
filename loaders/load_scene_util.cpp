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
  const float3 base_translation = optix::make_float3(scene_bbox_max[0] - scene_bbox_min[0], 
                                                     scene_bbox_max[1] - scene_bbox_min[1], 
                                                     0.0f);
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

