#include "load_scene.h"
#include "load_scene_util.h"
#include "../bake_api.h"

#include <vector_types.h>
#include <optixu/optixu_matrix_namespace.h>

#include <cfloat>
#include <iostream>


#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

namespace {
  struct ObjSceneMemory : public SceneMemory
  {
    ObjSceneMemory() {}
    virtual ~ObjSceneMemory() {}
  
    tinyobj::mesh_t obj_mesh;
    std::vector<bake::Mesh> meshes;
    std::vector<bake::Instance> instances;
  };

}   //namespace


bool load_obj_scene( const char* filename, bake::Scene& scene, float scene_bbox_min[3], float scene_bbox_max[3], SceneMemory*& base_memory, size_t num_instances_per_mesh )
{
  std::string errs;
  ObjSceneMemory* memory = new ObjSceneMemory();
  bool loaded = tinyobj::LoadObj(memory->obj_mesh, errs, filename);
  if (!errs.empty() || !loaded) {
    std::cerr << errs << std::endl;
    delete memory;
    return false;
  }

  memory->meshes.resize(1);
  bake::Mesh& mesh = memory->meshes[0];
  tinyobj::mesh_t& obj_mesh = memory->obj_mesh;

  mesh.num_vertices  = obj_mesh.positions.size()/3;
  mesh.num_triangles = obj_mesh.indices.size()/3;
  mesh.vertices      = &obj_mesh.positions[0];
  mesh.vertex_stride_bytes = 0;
  mesh.normals       = obj_mesh.normals.empty() ? NULL : &obj_mesh.normals[0];
  mesh.normal_stride_bytes = 0;
  mesh.tri_vertex_indices = &obj_mesh.indices[0];

  // Build bbox for mesh

  std::fill(mesh.bbox_min, mesh.bbox_min+3, FLT_MAX);
  std::fill(mesh.bbox_max, mesh.bbox_max+3, -FLT_MAX);
  for (size_t i = 0; i < mesh.num_vertices; ++i) {
    expand_bbox(mesh.bbox_min, mesh.bbox_max, &mesh.vertices[3*i]);
  }

  // Make instance

  memory->instances.resize(1);
  bake::Instance& instance = memory->instances[0];
  instance.mesh_index = 0;
  instance.storage_identifier = 0;
  const optix::Matrix4x4 mat = optix::Matrix4x4::identity();
  const float* matdata = mat.getData();
  std::copy(matdata, matdata+16, instance.xform);
  
  xform_bbox(mat, mesh.bbox_min, mesh.bbox_max, instance.bbox_min, instance.bbox_max);
  for (size_t k = 0; k < 3; ++k) {
    scene_bbox_min[k] = instance.bbox_min[k];
    scene_bbox_max[k] = instance.bbox_max[k];
  }

  if (num_instances_per_mesh > 1) {
    make_debug_instances(memory->meshes, memory->instances, num_instances_per_mesh-1, scene_bbox_min, scene_bbox_max);
  }

  scene.meshes = &memory->meshes[0];
  scene.num_meshes = memory->meshes.size();
  scene.instances = &memory->instances[0];
  scene.num_instances = memory->instances.size();
  base_memory = memory;
  return true;
}

