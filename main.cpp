//
// Copyright (c) 2015 NVIDIA Corporation.  All rights reserved.
// 
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto.  Any use, reproduction, disclosure or distribution of
// this software and related documentation without an express license agreement
// from NVIDIA Corporation is strictly prohibited.
// 
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF
// MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL
// NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR
// CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR
// LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF BUSINESS
// INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
// INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGES
//


#include "bake_api.h"
#include "bake_view.h"
#include "bake_util.h"

#include <main.h>
#include <optixu/optixu_matrix_namespace.h>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <set>

const size_t NUM_RAYS = 64;
const size_t SAMPLES_PER_FACE = 3;


// For parsing command line into constants
struct Config {
  std::vector<std::string> obj_filenames;
  size_t num_instances_per_mesh;
  int num_samples;
  int min_samples_per_face;
  int num_rays;
  bake::VertexFilterMode filter_mode;
  float regularization_weight;
  bool use_ground_plane_blocker;

  Config( int argc, const char ** argv ) {
    // set defaults
    num_instances_per_mesh = 1;
    num_samples = 0;  // default means determine from mesh
    min_samples_per_face = SAMPLES_PER_FACE;
    num_rays    = NUM_RAYS; 
#ifdef EIGEN3_ENABLED
    filter_mode = bake::VERTEX_FILTER_LEAST_SQUARES;
#else
    filter_mode = bake::VERTEX_FILTER_AREA_BASED;
#endif
    regularization_weight = 0.1f;
    use_ground_plane_blocker = true;

    // parse arguments
    for ( int i = 1; i < argc; ++i ) 
    { 
      std::string arg( argv[i] );
      if( arg == "-h" || arg == "--help" ) 
      {
        printUsageAndExit( argv[0] ); 
      } 
      else if( (arg == "-o" || arg == "--obj") && i+1 < argc ) 
      {
        std::string str = argv[++i];
        obj_filenames.push_back(str);
      } 
      else if ( (arg == "-i" || arg == "--instances") && i+1 < argc )
      {
        int n = -1;
        if( (sscanf( argv[++i], "%d", &n ) != 1) || n < 1 ) {
          printParseErrorAndExit( argv[0], arg, argv[i] );
        }
        num_instances_per_mesh = static_cast<size_t>(n);
      }
      else if ( (arg == "-s" || arg == "--samples") && i+1 < argc )
      {
        if( sscanf( argv[++i], "%d", &num_samples ) != 1 ) {
          printParseErrorAndExit( argv[0], arg, argv[i] );
        }
      }
      else if ( (arg == "-t" || arg == "--samples_per_face") && i+1 < argc )
      {
        if (sscanf( argv[++i], "%d", &min_samples_per_face ) != 1 ) {
          printParseErrorAndExit( argv[0], arg, argv[i] );
        }

      }
      else if ( (arg == "-r" || arg == "--rays") && i+1 < argc )
      {
        if( sscanf( argv[++i], "%d", &num_rays ) != 1 ) {
          printParseErrorAndExit( argv[0], arg, argv[i] );
        }
      }
      else if ( (arg == "--no_ground_plane" ) ) {
        use_ground_plane_blocker = false;
      }
      else if ( (arg == "--no_least_squares" ) ) {
        filter_mode = bake::VERTEX_FILTER_AREA_BASED;  
      }
      else if ( (arg == "-w" || arg == "--regularization_weight" ) && i+1 < argc ) {
        if( sscanf( argv[++i], "%f", &regularization_weight ) != 1 ) {
          printParseErrorAndExit( argv[0], arg, argv[i] );
        }
        regularization_weight = std::max( regularization_weight, 0.0f );
      }
      else 
      {
        std::cerr << "Bad option: '" << arg << "'" << std::endl;
        printUsageAndExit( argv[0] );
      }
    }

    if (obj_filenames.empty()) {
      // default filename
#ifdef PROJECT_ABSDIRECTORY
      obj_filenames.push_back(std::string(PROJECT_ABSDIRECTORY) + std::string("/assets/lucy.obj"));
#else
      obj_filenames.push_back(std::string( "./assets/lucy.obj" ));
#endif
    }
  }

  void printUsageAndExit( const char* argv0 )
  {
    std::cerr
    << "Usage  : " << argv0 << " [options]\n"
    << "App options:\n"
    << "  -h  | --help                          Print this usage message\n"
    << "  -o  | --obj <obj_file>                Specify model to be rendered.  Repeat this flag for multiple models.\n"
    << "  -i  | --instances <n>                 Number of instances per mesh (default 1).  For testing.\n"
    << "  -r  | --rays    <n>                   Number of rays per sample point for gather (default " << NUM_RAYS << ")\n"
    << "  -s  | --samples <n>                   Number of sample points on mesh (default " << SAMPLES_PER_FACE << " per face; any extra samples are based on area)\n"
    << "  -t  | --samples_per_face <n>          Minimum number of samples per face (default " << SAMPLES_PER_FACE << ")\n"
    << "        --no_ground_plane               Disable virtual XZ ground plane\n"
#ifdef EIGEN3_ENABLED
    << "  -w  | --regularization_weight <w>     Regularization weight for least squares, positive range. (default 0.1)\n"
    << "        --no_least_squares              Disable least squares filtering\n"
#endif
    << std::endl;
    
    exit(1);
  }
  
  void printParseErrorAndExit( const char* argv0, const std::string& flag, const char* arg )
  {
    std::cerr << "Could not parse argument: " << flag << " " << arg << std::endl;
    printUsageAndExit( argv0 );
  }


};

namespace {

  void xform_bbox(const optix::Matrix4x4& mat, const float in_min[3], const float in_max[3],
                  float out_min[3], float out_max[3])
  {
    float4 a = mat*optix::make_float4( in_min[0], in_min[1], in_min[2], 1.0f);
    float4 b = mat*optix::make_float4( in_max[0], in_max[1], in_max[2], 1.0f);
    for (size_t k = 0; k < 3; ++k) {
      out_min[k] = (&a.x)[k];
      out_max[k] = (&b.x)[k];
    }
  }

  void expand_bbox(float bbox_min[3], float bbox_max[3], float v[3])
  {
    for (size_t k = 0; k < 3; ++k) {
      bbox_min[k] = std::min(bbox_min[k], v[k]);
      bbox_max[k] = std::max(bbox_max[k], v[k]);
    }
  }

  void make_ground_plane(float scene_bbox_min[3], float scene_bbox_max[3], unsigned scene_vertex_stride_bytes,
                         std::vector<float>& plane_vertices, std::vector<unsigned int>& plane_indices,
                         std::vector<bake::Mesh>& meshes, std::vector<bake::Instance>& instances)
  {

    const unsigned int index_data[] = {0, 1, 2, 0, 2, 3};
    plane_indices.resize(6);
    std::copy(index_data, index_data+6, plane_indices.begin());
    float scene_extents[] = {scene_bbox_max[0] - scene_bbox_min[0],
                             scene_bbox_max[1] - scene_bbox_min[1],
                             scene_bbox_max[2] - scene_bbox_min[2]};
    const float scale_factor = 100.0f;
    float ground_min[] = {scene_bbox_max[0] - scale_factor*scene_extents[0],
                          scene_bbox_min[1],
                          scene_bbox_max[2] - scale_factor*scene_extents[2]};
    float ground_max[] = {scene_bbox_min[0] + scale_factor*scene_extents[0],
                          scene_bbox_min[1],
                          scene_bbox_min[2] + scale_factor*scene_extents[2]};
    const float vertex_data[] = {ground_min[0], ground_min[1], ground_min[2],
                                 ground_max[0], ground_min[1], ground_min[2],
                                 ground_max[0], ground_min[1], ground_max[2],
                                 ground_min[0], ground_min[1], ground_max[2]};

    // OptiX Prime requires all meshes in the same scene to have the same vertex stride.
    const unsigned vertex_stride_bytes = scene_vertex_stride_bytes > 0 ? scene_vertex_stride_bytes : 3*sizeof(float);
    assert(vertex_stride_bytes % sizeof(float) == 0);
    const unsigned num_floats_per_vert = vertex_stride_bytes / sizeof(float);
    plane_vertices.resize(4*(num_floats_per_vert));
    std::fill(plane_vertices.begin(), plane_vertices.end(), 0.0f);
    for (size_t i = 0; i < 4; ++i) {
      plane_vertices[num_floats_per_vert*i  ] = vertex_data[3*i];
      plane_vertices[num_floats_per_vert*i+1] = vertex_data[3*i+1];
      plane_vertices[num_floats_per_vert*i+2] = vertex_data[3*i+2];
    }
    
    bake::Mesh plane_mesh;
    plane_mesh.num_vertices  = 4;
    plane_mesh.num_triangles = 2;
    plane_mesh.vertices      = &plane_vertices[0];
    plane_mesh.vertex_stride_bytes = vertex_stride_bytes;
    plane_mesh.normals       = NULL;
    plane_mesh.normal_stride_bytes = 0;
    plane_mesh.tri_vertex_indices = &plane_indices[0];
    
    bake::Instance instance;
    instance.mesh_index = meshes.size();

    const optix::Matrix4x4 mat = optix::Matrix4x4::identity();
    const float* matdata = mat.getData();
    std::copy(matdata, matdata+16, instance.xform);
    
    for (size_t k = 0; k < 3; ++k) {
      instance.bbox_min[k] = ground_min[k];
      instance.bbox_max[k] = ground_max[k];
    }

    meshes.push_back(plane_mesh);
    instances.push_back(instance);

  }

  // TODO: this should operate on entire scene
  void make_debug_instances(bake::Mesh& bake_mesh, unsigned mesh_index, size_t n, std::vector<bake::Instance>& instances,
                            float scene_bbox_min[3], float scene_bbox_max[3])
  {

    // Make up a transform per instance
    const float3 bbox_base = optix::make_float3(0.5f*(bake_mesh.bbox_min[0] + bake_mesh.bbox_max[0]),
                                                      bake_mesh.bbox_min[1],
                                                0.5f*(bake_mesh.bbox_min[2] + bake_mesh.bbox_max[2]));
    const float rot = 0.5236f;  // pi/6
    const float3 rot_axis = optix::make_float3(0.0f, 1.0f, 0.0f);
    const float scale_factor = 0.9f;
    float scale = scale_factor;
    const float3 base_translation = 1.01*optix::make_float3(bake_mesh.bbox_max[0] - bake_mesh.bbox_min[0], 0.0f, 0.0f);
    float3 translation = scale_factor* base_translation;

    for (size_t i = 0; i < n; i++) {
      bake::Instance instance;
      instance.mesh_index = mesh_index;
      const optix::Matrix4x4 mat = optix::Matrix4x4::translate(translation) *
                                   optix::Matrix4x4::translate(bbox_base) *
                                   optix::Matrix4x4::rotate((i+1)*rot, rot_axis) *
                                   optix::Matrix4x4::scale(optix::make_float3(scale)) *
                                   optix::Matrix4x4::translate(-bbox_base);
      scale *= scale_factor;
      translation += scale*base_translation;

      xform_bbox(mat, bake_mesh.bbox_min, bake_mesh.bbox_max, instance.bbox_min, instance.bbox_max);
      expand_bbox(scene_bbox_min, scene_bbox_max, instance.bbox_min);
      expand_bbox(scene_bbox_min, scene_bbox_max, instance.bbox_max);

      const float* matdata = mat.getData();
      std::copy(matdata, matdata+16, instance.xform);
      instances.push_back(instance);
    }
  }

  void allocate_ao_samples(bake::AOSamples& ao_samples, unsigned int n) {
    ao_samples.num_samples = n;
    ao_samples.sample_positions = new float[3*n];
    ao_samples.sample_normals = new float[3*n];
    ao_samples.sample_face_normals = new float[3*n];
    ao_samples.sample_infos = new bake::SampleInfo[n];
  }

  void destroy_ao_samples(bake::AOSamples& ao_samples) {
    delete [] ao_samples.sample_positions;
    ao_samples.sample_positions = NULL;
    delete [] ao_samples.sample_normals;
    ao_samples.sample_normals = NULL;
    delete [] ao_samples.sample_face_normals;
    ao_samples.sample_face_normals = NULL;
    delete [] ao_samples.sample_infos;
    ao_samples.sample_infos = NULL;
    ao_samples.num_samples = 0;
  }

}


#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"


// Required entry point
//------------------------------------------------------------------------------
int sample_main( int argc, const char** argv )
{
  
  // show console and redirect printing
  NVPWindow::sysVisibleConsole();

  const Config config( argc, argv ); 
  
  Timer timer;

  //
  // Load model file
  //
  std::cerr << "Load meshes ...              "; std::cerr.flush();

  timer.start();
  std::vector<tinyobj::mesh_t> meshes;
  for(size_t i = 0; i < config.obj_filenames.size(); ++i) { 
    std::string errs;
    tinyobj::mesh_t mesh;
    bool loaded = tinyobj::LoadObj(mesh, errs, config.obj_filenames[i].c_str());
    if (!errs.empty() || !loaded) {
      std::cerr << errs << std::endl;
      return 0; 
    }
    meshes.push_back(mesh);
  }

  printTimeElapsed( timer ); 

  std::cerr << "Mesh statistics:" << std::endl;
  for (size_t i = 0; i < config.obj_filenames.size(); ++i) {
    const tinyobj::mesh_t& mesh = meshes[i];
    std::cerr << config.obj_filenames[i] << ": " << std::endl << "\t"
              << mesh.positions.size()/3 << " vertices, "
              << mesh.normals.size()/3 << " normals, " 
              << mesh.indices.size()/3 << " triangles" << std::endl;
  }
  
  std::cerr << "Minimum samples per face: " << config.min_samples_per_face << std::endl;

  //
  // Populate instances
  //
  std::vector<bake::Mesh> bake_meshes(meshes.size());
  const size_t num_instances = meshes.size() * config.num_instances_per_mesh;
  std::vector<bake::Instance> instances;
  instances.reserve(num_instances);

  float scene_bbox_min[] = {FLT_MAX, FLT_MAX, FLT_MAX};
  float scene_bbox_max[] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
  for (size_t meshIdx = 0; meshIdx < meshes.size(); ++meshIdx) {
    
    tinyobj::mesh_t& mesh = meshes[meshIdx];
    bake::Mesh& bake_mesh = bake_meshes[meshIdx];
    bake_mesh.num_vertices  = mesh.positions.size()/3;
    bake_mesh.num_triangles = mesh.indices.size()/3;
    bake_mesh.vertices      = &mesh.positions[0];
    bake_mesh.vertex_stride_bytes = 0;
    bake_mesh.normals       = mesh.normals.empty() ? NULL : &mesh.normals[0];
    bake_mesh.normal_stride_bytes = 0;
    bake_mesh.tri_vertex_indices = &mesh.indices[0];

    // Build bbox for mesh

    std::fill(bake_mesh.bbox_min, bake_mesh.bbox_min+3, FLT_MAX);
    std::fill(bake_mesh.bbox_max, bake_mesh.bbox_max+3, -FLT_MAX);
    for (size_t i = 0; i < mesh.positions.size()/3; ++i) {
      expand_bbox(bake_mesh.bbox_min, bake_mesh.bbox_max, &mesh.positions[3*i]);
    }

    // Make instance

    bake::Instance instance;
    instance.mesh_index = meshIdx;
    const optix::Matrix4x4 mat = optix::Matrix4x4::identity();  // TODO: use transform from file
    const float* matdata = mat.getData();
    std::copy(matdata, matdata+16, instance.xform);
    
    xform_bbox(mat, bake_mesh.bbox_min, bake_mesh.bbox_max, instance.bbox_min, instance.bbox_max);
    expand_bbox(scene_bbox_min, scene_bbox_max, instance.bbox_min);
    expand_bbox(scene_bbox_min, scene_bbox_max, instance.bbox_max);

    instances.push_back(instance);

    if (config.num_instances_per_mesh > 1) {
      make_debug_instances(bake_mesh, meshIdx, config.num_instances_per_mesh-1, instances, scene_bbox_min, scene_bbox_max);
    }
    
  }


  assert(instances.size() == num_instances);

  // OptiX Prime requires all instances to have the same vertex stride
  for (size_t i = 1; i < bake_meshes.size(); ++i) {
    if (bake_meshes[i].vertex_stride_bytes != bake_meshes[0].vertex_stride_bytes) {
      std::cerr << "Error: all meshes must have the same vertex stride.  Bailing.\n";
      exit(-1);
    }
  }

  //
  // Generate AO samples
  //

  std::cerr << "Generate sample points ... \n"; std::cerr.flush();

  timer.reset();
  timer.start();

  bake::Scene scene;
  scene.meshes = &bake_meshes[0];
  scene.num_meshes = bake_meshes.size();
  scene.instances = &instances[0];
  scene.num_instances = instances.size();

  std::vector<size_t> num_samples_per_instance(num_instances);
  const size_t total_samples = bake::distributeSamples( scene, config.min_samples_per_face, config.num_samples, &num_samples_per_instance[0] );

  bake::AOSamples ao_samples;
  allocate_ao_samples( ao_samples, total_samples );

  bake::sampleInstances( scene, &num_samples_per_instance[0], config.min_samples_per_face, ao_samples );
  
  printTimeElapsed( timer ); 

  std::cerr << "Total samples: " << total_samples << std::endl;

  //
  // Evaluate AO samples 
  //
  std::cerr << "Compute AO ...             "; std::cerr.flush();
  
  timer.reset();
  timer.start();

  std::vector<float> ao_values( total_samples );
  std::fill( ao_values.begin(), ao_values.end(), 0.0f );

  if (config.use_ground_plane_blocker) {
    // Add blocker for ground plane (no surface samples)
    std::vector<bake::Mesh> blocker_meshes;
    std::vector<bake::Instance> blocker_instances;
    std::vector<float> plane_vertices;
    std::vector<unsigned int> plane_indices;
    make_ground_plane(scene_bbox_min, scene_bbox_max, bake_meshes[0].vertex_stride_bytes, 
      plane_vertices, plane_indices, blocker_meshes, blocker_instances);
    bake::Scene blockers = { &blocker_meshes[0], blocker_meshes.size(), &blocker_instances[0], blocker_instances.size() };
    bake::computeAOWithBlockers(scene, blockers,
      ao_samples, config.num_rays, scene_bbox_min, scene_bbox_max, &ao_values[0] );
  } else {
    bake::computeAO( scene, ao_samples, config.num_rays, scene_bbox_min, scene_bbox_max, &ao_values[0] );
  }
  printTimeElapsed( timer ); 

  std::cerr << "Map AO to vertices  ...    "; std::cerr.flush();

  timer.reset();
  timer.start();
  float** vertex_ao = new float*[ num_instances ];
  for (size_t i = 0; i < num_instances; ++i ) {
    vertex_ao[i] = new float[ bake_meshes[instances[i].mesh_index].num_vertices ];
  }
  bake::mapAOToVertices( scene, &num_samples_per_instance[0], ao_samples, &ao_values[0], config.filter_mode, config.regularization_weight, vertex_ao );

  printTimeElapsed( timer ); 

  //
  // Visualize results
  //
  std::cerr << "Launch viewer  ... \n" << std::endl;
  bake::view( &bake_meshes[0], bake_meshes.size(), &instances[0], instances.size(), vertex_ao, scene_bbox_min, scene_bbox_max );

  for (size_t i = 0; i < instances.size(); ++i) {
    delete [] vertex_ao[i];
  }
  delete [] vertex_ao;

  destroy_ao_samples( ao_samples );
  
  return 1;
}

// Required logging function
void sample_print(int level, const char * fmt)
{
  //stub
}

