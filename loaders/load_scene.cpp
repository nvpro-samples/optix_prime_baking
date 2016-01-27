
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
  } else if (extension == ".bk3d") {
    //return load_bk3d_scene(filename, scene, scene_bbox_min, scene_bbox_max, memory, num_instances_per_mesh);
  }

  std::cerr << "Unrecognized filename extension: " << extension << std::endl;
  return false;
}
