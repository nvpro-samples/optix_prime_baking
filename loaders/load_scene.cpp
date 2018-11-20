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

#include "load_scene.h"
#include <iostream>
#include <string>

bool load_scene( const char* filename, bake::Scene& scene, float* scene_bbox_min, float* scene_bbox_max, SceneMemory*& memory, size_t num_instances_per_mesh )
{
  if (!filename) return false;
  
  std::string s(filename);
  if (s.length() < 4) return false;

  size_t pos = s.rfind(".");
  if (pos == std::string::npos) {
    std::cerr << "Could not parse filename extension for: " << filename << std::endl;
    return false;
  }

  std::string extension = s.substr(pos);
  if (extension == ".obj") {
    return load_obj_scene(filename, scene, scene_bbox_min, scene_bbox_max, memory, num_instances_per_mesh);
  }
  
#ifdef NOGZLIB
  if (extension == ".gz") {
    std::cerr << "Unhandled .gz extension; must rebuild with ZLIB support to load this file." << std::endl;
    std::cerr << filename << std::endl;
    return false;
  }
#endif
  if (extension == ".gz" && pos > 1) {
    size_t prepos = s.rfind(".", pos-1);
    if (prepos != std::string::npos) {
      extension = s.substr(prepos);
    }
  }

  if (extension == ".csf" || extension == ".csf.gz") {
    return load_csf_scene(filename, scene, scene_bbox_min, scene_bbox_max, memory, num_instances_per_mesh);
  }

  if (extension == ".bk3d" || extension == ".bk3d.gz") {
    return load_bk3d_scene(filename, scene, scene_bbox_min, scene_bbox_max, memory, num_instances_per_mesh);
  }

  std::cerr << "Unhandled filename extension: " << extension << ".  Attempting to load as bk3d" << std::endl;
  std::cerr << filename << std::endl;
  return load_bk3d_scene(filename, scene, scene_bbox_min, scene_bbox_max, memory, num_instances_per_mesh);
}
