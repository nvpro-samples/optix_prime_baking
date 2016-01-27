#include "load_scene.h"
#include "load_scene_util.h"
#include "../bake_api.h"

#include "bk3dEx.h"

#include <vector_types.h>
#include <optixu/optixu_matrix_namespace.h>

#include <cfloat>
#include <iostream>
#include <sstream>
#include <stdexcept>

class AssertException : public std::runtime_error
{
public:
  AssertException( const char* file, int line, const char* condition )
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
  struct Bk3dSceneMemory : public SceneMemory
  {
    Bk3dSceneMemory(bk3d::FileHeader* bk3dHeader, void* bk3dBuffer)
      : bk3dHeader(bk3dHeader), bk3dBuffer(bk3dBuffer)
    {
    }
    virtual ~Bk3dSceneMemory()
    {
      free(bk3dHeader);
      free(bk3dBuffer);
    }

    bk3d::FileHeader *bk3dHeader;
    void* bk3dBuffer;
  
    std::vector<bake::Mesh> meshes;
    std::vector<bake::Instance> instances;
  };

}   //namespace


bool load_bk3d_scene( const char* filename, bake::Scene& scene, float scene_bbox_min[3], float scene_bbox_max[3], SceneMemory*& base_memory, size_t num_instances_per_mesh )
{
  void * pBk3dBufferMemory = NULL;
  unsigned int bk3dBufferMemorySz = 0;
  bk3d::FileHeader* bk3dData = bk3d::load(filename, &pBk3dBufferMemory, &bk3dBufferMemorySz);
  if (!bk3dData) return false;

  Bk3dSceneMemory* memory = new Bk3dSceneMemory(bk3dData, pBk3dBufferMemory);

  assert(bk3dData->pMeshes);
  memory->meshes.reserve(bk3dData->pMeshes->n);
  memory->instances.reserve(bk3dData->pMeshes->n);  // will have at least as many instances as meshes

  std::fill(scene_bbox_min, scene_bbox_min+3, FLT_MAX);
  std::fill(scene_bbox_max, scene_bbox_max+3, -FLT_MAX);

  for (int meshIdx = 0; meshIdx < bk3dData->pMeshes->n; ++meshIdx) {
    // Assumptions (enforced with assertions):
    // - vertices are float
    // - vertices are in Attribute 0
    // - normals are in Attribute 1
    // - indices are unsigned int format (could be unsigned short etc.)
    // - bk3d file doesn't give separate indices for attributes

    // Make separate mesh/instance per prim group
    bk3d::Mesh* pMesh = bk3dData->pMeshes->p[meshIdx];

    bool skip_mesh = true;
    for (int pg = 0; pg < pMesh->pPrimGroups->n; pg++) {
      bk3d::PrimGroup* pPG = pMesh->pPrimGroups->p[pg];
      if (pPG->topologyGL == GL_TRIANGLES) {
        skip_mesh = false;
        break;
      }
    }
    if (skip_mesh) continue;

    // get the slot where the vertex pos is located (maybe mixed with other attributes)

    RT_ASSERT( pMesh->pAttributes->n >= 1 && "Mesh must have position attribute" );

    bk3d::Attribute* pPositionAttr = pMesh->pAttributes->p[0];
    RT_ASSERT( pPositionAttr->formatGL == GL_FLOAT && "Mesh must have vertex positions of type float" );

    bk3d::Slot* pPositionSlot = pMesh->pSlots->p[pPositionAttr->slot];
    float* vertices = (float*)pPositionAttr->pAttributeBufferData;
    const size_t num_vertices = pPositionSlot->vertexCount;
    const unsigned vertex_stride_bytes = pPositionAttr->strideBytes;

    float* normals = NULL;
    unsigned normal_stride_bytes = 0;
    if (pMesh->pAttributes->n > 1) {
      bk3d::Attribute* pNormalAttr = pMesh->pAttributes->p[1];
      RT_ASSERT( pNormalAttr->formatGL == GL_FLOAT && "Mesh must have normals of type float" );

      bk3d::Slot* pNormalSlot = pMesh->pSlots->p[pNormalAttr->slot];
      normals = (float*)pNormalAttr->pAttributeBufferData;
      normal_stride_bytes = pNormalAttr->strideBytes;
    }

    optix::Matrix4x4 mesh_xform = optix::Matrix4x4::identity();
    if (pMesh->pTransforms && pMesh->pTransforms->n > 0) {
      mesh_xform = optix::Matrix4x4(pMesh->pTransforms->p[0]->Matrix().m);
      mesh_xform.transpose();  // OptiX matrices are transposed from OpenGL/bk3d
    }

    for (int pg = 0; pg < pMesh->pPrimGroups->n; pg++) {
      bk3d::PrimGroup* pPG = pMesh->pPrimGroups->p[pg];
      if(pPG->topologyGL != GL_TRIANGLES) continue;

      RT_ASSERT( pPG->indexFormatGL == GL_UNSIGNED_INT );

      bake::Mesh bake_mesh;

      // Same vertex buffer for each primgroup. Prime can also share these.
      bake_mesh.num_vertices  = num_vertices;
      bake_mesh.vertices      = vertices;
      bake_mesh.vertex_stride_bytes = vertex_stride_bytes;

      bake_mesh.normals       = normals;
      bake_mesh.normal_stride_bytes = normal_stride_bytes;

      bake_mesh.num_triangles = pPG->primitiveCount;
      bake_mesh.tri_vertex_indices = (unsigned int*)pPG->pIndexBufferData;

      bool compute_bbox = false;
      for (size_t k = 0; k < 3; ++k) {
        bake_mesh.bbox_min[k] = pPG->aabbox.min[k];
        bake_mesh.bbox_max[k] = pPG->aabbox.max[k];
        if (bake_mesh.bbox_min[k] > bake_mesh.bbox_max[k]) {
          compute_bbox = true;
          break;
        }
      }

      if (compute_bbox) {
        // Bbox stored in file is empty, so compute from vertices.
        std::fill(bake_mesh.bbox_min, bake_mesh.bbox_min+3, FLT_MAX);
        std::fill(bake_mesh.bbox_max, bake_mesh.bbox_max+3, -FLT_MAX);
        for (size_t i = 0; i < bake_mesh.num_vertices; ++i) {
          expand_bbox(bake_mesh.bbox_min, bake_mesh.bbox_max, &bake_mesh.vertices[3*i]);
        }
      }

      bake::Instance instance;
      instance.mesh_index = memory->meshes.size();

      optix::Matrix4x4 group_xform = optix::Matrix4x4::identity();
      if (pPG->pTransforms && pPG->pTransforms->n > 0) {
        group_xform = optix::Matrix4x4(pPG->pTransforms->p[0]->Matrix().m);
        group_xform.transpose();
      }
      optix::Matrix4x4 mat = mesh_xform * group_xform;
      std::copy(mat.getData(), mat.getData()+16, instance.xform);
      xform_bbox(mat, bake_mesh.bbox_min, bake_mesh.bbox_max, instance.bbox_min, instance.bbox_max);

      expand_bbox(scene_bbox_min, scene_bbox_max, instance.bbox_min);
      expand_bbox(scene_bbox_min, scene_bbox_max, instance.bbox_max);

      memory->meshes.push_back(bake_mesh);
      memory->instances.push_back(instance);
    }
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


