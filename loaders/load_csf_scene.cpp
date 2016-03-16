#include "load_scene.h"
#include "load_scene_util.h"
#include "../bake_api.h"

#include "cadscenefile.h"

#include <vector_types.h>
#include <optixu/optixu_matrix_namespace.h>

#include <cfloat>
#include <iostream>
#include <sstream>
#include <stdexcept>

class AssertException : public std::runtime_error
{
public:
  AssertException(const char* file, int line, const char* condition)
    : std::runtime_error("Assertion Failed")
  {
    std::ostringstream ss;
    ss << "Assertion Failed: " << file << ":" << line << ": " << condition;
    msg = ss.str();
  }
  virtual ~AssertException() throw() {}
  virtual const char* what() const throw() { return msg.c_str(); }
private:
  std::string msg;
};

// Runtime assertion that still works in Release mode
#define RT_ASSERT(condition)                                                       \
  do {                                                                             \
    if(!(condition))                                                               \
      throw AssertException( __FILE__, __LINE__, #condition );                     \
      } while (0)

namespace {
  struct CSFSceneMemory : public SceneMemory
  {
    CSFSceneMemory(CSFile* file, CSFileMemoryPTR mem)
      : csf(file), csfMem(mem)
    {
    }
    virtual ~CSFSceneMemory()
    {
      CSFileMemory_delete(csfMem);
    }

    CSFile *csf;
    CSFileMemoryPTR csfMem;

    std::vector<bake::Mesh> meshes;
    std::vector<bake::Instance> instances;

  };

}   //namespace


bool load_csf_scene(const char* filename, bake::Scene& scene, float scene_bbox_min[3], float scene_bbox_max[3], SceneMemory*& base_memory, size_t num_instances_per_mesh)
{
  CSFileMemoryPTR csfMem = CSFileMemory_new();
  CSFile* csf;
  
  if (CSFile_loadExt(&csf, filename, csfMem) != CADSCENEFILE_NOERROR || (csf->fileFlags & CADSCENEFILE_FLAG_STRIPS)) {
    CSFileMemory_delete(csfMem);
    return false;
  }

  CSFile_transform(csf);

  CSFSceneMemory* memory = new CSFSceneMemory(csf, csfMem);

  std::vector<int>   referencedGeometry(csf->numGeometries, 0);

  // for debugging large models
//#define GEOMETRY_LOWER_LIMIT  13400
//#define GEOMETRY_UPPER_LIMIT  17000

  int numObjects = 0;
  for (int n = 0; n < csf->numNodes; n++){
    CSFNode* node = csf->nodes + n;

    if (node->geometryIDX < 0) continue;
#if defined(GEOMETRY_UPPER_LIMIT) && defined(GEOMETRY_LOWER_LIMIT)
    if (node->geometryIDX < GEOMETRY_LOWER_LIMIT  || node->geometryIDX > GEOMETRY_UPPER_LIMIT) continue;
#endif
    for (int p = 0; p < node->numParts; p++){
      if (node->parts[p].active){
        referencedGeometry[node->geometryIDX] = 1;
        numObjects++;
        break;
      }
    }
  }
  
  memory->meshes.reserve(csf->numGeometries);
  memory->instances.reserve(numObjects);

  std::fill(scene_bbox_min, scene_bbox_min + 3, FLT_MAX);
  std::fill(scene_bbox_max, scene_bbox_max + 3, -FLT_MAX);

  for (int g = 0; g < csf->numGeometries; g++) {
    if (!referencedGeometry[g]) continue;

    // Make separate mesh/instance per prim group
    CSFGeometry* geom = csf->geometries + g;

    //if (!referencedGeometry[g]) continue; 
    // for simplicity just add all geometry, even if unreferenced

    bake::Mesh bake_mesh;

    // Same vertex buffer for each primgroup. Prime can also share these.
    bake_mesh.num_vertices = geom->numVertices;
    bake_mesh.vertices = geom->vertex;
    bake_mesh.vertex_stride_bytes = sizeof(float) * 3;

    bake_mesh.normals = geom->normal;
    bake_mesh.normal_stride_bytes = sizeof(float) * 3;

    bake_mesh.num_triangles = geom->numIndexSolid / 3;
    bake_mesh.tri_vertex_indices = geom->indexSolid;

    {
      // compute bbox
      std::fill(bake_mesh.bbox_min, bake_mesh.bbox_min + 3, FLT_MAX);
      std::fill(bake_mesh.bbox_max, bake_mesh.bbox_max + 3, -FLT_MAX);
      unsigned char* p = reinterpret_cast<unsigned char*>(bake_mesh.vertices);
      for (size_t i = 0; i < bake_mesh.num_vertices; ++i) {
        expand_bbox(bake_mesh.bbox_min, bake_mesh.bbox_max, reinterpret_cast<float*>(p));
        p += sizeof(float) * 3;
      }
    }

    referencedGeometry[g] = int(memory->meshes.size());

    memory->meshes.push_back(bake_mesh);
  }

  for (int n = 0; n < csf->numNodes; n++){
    CSFNode* node = csf->nodes + n;

    if (node->geometryIDX < 0) continue;
#if defined(GEOMETRY_UPPER_LIMIT) && defined(GEOMETRY_LOWER_LIMIT)
    if (node->geometryIDX < GEOMETRY_LOWER_LIMIT || node->geometryIDX > GEOMETRY_UPPER_LIMIT) continue;
#endif
    // fixme, for simplicity we currently don't deal with individual part baking
    bool active = false;
    for (int p = 0; p < node->numParts; p++){
      if (node->parts[p].active){
        active = true;
        break;
      }
    }

    if (!active) continue;

    bake::Instance instance;
    instance.mesh_index = referencedGeometry[node->geometryIDX];
    instance.storage_identifier = n;

    bake::Mesh& bake_mesh = memory->meshes[node->geometryIDX];

    optix::Matrix4x4 xform = optix::Matrix4x4(node->worldTM).transpose();
    std::copy(xform.getData(), xform.getData() + 16, instance.xform);

    xform_bbox(xform, bake_mesh.bbox_min, bake_mesh.bbox_max, instance.bbox_min, instance.bbox_max);
    expand_bbox(scene_bbox_min, scene_bbox_max, instance.bbox_min);
    expand_bbox(scene_bbox_min, scene_bbox_max, instance.bbox_max);

    memory->instances.push_back(instance);
  }

  scene.meshes = &memory->meshes[0];
  scene.num_meshes = memory->meshes.size();
  scene.instances = &memory->instances[0];
  scene.num_instances = memory->instances.size();
  base_memory = memory;

  return true;
}


