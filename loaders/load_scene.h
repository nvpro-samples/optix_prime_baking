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

#include <cstddef>

struct SceneMemory {
  virtual ~SceneMemory() { }
};

// forward decl
namespace bake {
  struct Scene;
}

bool load_obj_scene(const char* filename, bake::Scene& scene, float* scene_bbox_min, float* scene_bbox_max, SceneMemory*& memory, size_t num_instances_per_mesh=1 );
bool load_bk3d_scene(const char* filename, bake::Scene& scene, float* scene_bbox_min, float* scene_bbox_max, SceneMemory*& memory, size_t num_instances_per_mesh=1 );
bool load_csf_scene(const char* filename, bake::Scene& scene, float* scene_bbox_min, float* scene_bbox_max, SceneMemory*& memory, size_t num_instances_per_mesh = 1);

// Choose one of the above based on filename
bool load_scene(const char* filename, bake::Scene& scene, float* scene_bbox_min, float* scene_bbox_max, SceneMemory*& memory, size_t num_instances_per_mesh=1 );

