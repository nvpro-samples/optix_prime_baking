//
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
#include "loaders/load_scene.h"

#include <main.h>
#include <optixu/optixu_matrix_namespace.h>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <set>
#include <sys/stat.h>

const size_t NUM_RAYS = 64;
const size_t SAMPLES_PER_FACE = 3;
const char* DEFAULT_BK3D_FILE = "sled_v134.bk3d.gz";
const char* DEFAULT_OBJ_FILE = "lucy.obj";
#ifdef PROJECT_ABSDIRECTORY
  #define ASSET_PATH PROJECT_ABSDIRECTORY "/assets/"
#else
  #define ASSET_PATH "./assets/"
#endif


// For parsing command line into constants
struct Config {
  std::string scene_filename;
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
      else if( (arg == "-f" || arg == "--file") && i+1 < argc ) 
      {
        assert( scene_filename.empty() && "multiple -f (--file) flags found when parsing command line");
        scene_filename = argv[++i];
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

    if (scene_filename.empty()) {

      // Make default filename

      const std::string asset_path(ASSET_PATH);

#ifndef NOGZLIB
      // Try bk3d file first
      std::string bk3d_path = asset_path + std::string(DEFAULT_BK3D_FILE);
      struct stat buf;
      if (stat(bk3d_path.c_str(), &buf) == 0) {
        scene_filename = bk3d_path;
      }
#endif
      if (scene_filename.empty()) {
        // Fall back to simpler obj file
        std::string obj_path = asset_path + std::string(DEFAULT_OBJ_FILE);
        struct stat buf;
        if (stat(obj_path.c_str(), &buf) == 0) {
          scene_filename = obj_path;
        } else {
          std::cerr << "Could not find default scene file (" << obj_path << "). Use options to specify one." << std::endl;
          printUsageAndExit( argv[0] );
        }
      }
     
    }
  }

  void printUsageAndExit( const char* argv0 )
  {
    std::cerr
    << "Usage  : " << argv0 << " [options]\n"
    << "App options:\n"
    << "  -h  | --help                          Print this usage message\n"
    << "  -f  | --file <scene_file>             Specify model to be rendered.\n"
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



// Required entry point
//------------------------------------------------------------------------------
int sample_main( int argc, const char** argv )
{
  
  // show console and redirect printing
  NVPWindow::sysVisibleConsole();

  const Config config( argc, argv ); 
  
  Timer timer;

  //
  // Load scene
  //
  std::cerr << "Load scene ...              "; std::cerr.flush();

  timer.start();

  bake::Scene scene;
  SceneMemory* scene_memory;
  float scene_bbox_min[] = {FLT_MAX, FLT_MAX, FLT_MAX};
  float scene_bbox_max[] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
  if (!load_scene( config.scene_filename.c_str(), scene, scene_bbox_min, scene_bbox_max, scene_memory, config.num_instances_per_mesh )) {
    std::cerr << "Failed to load scene, exiting" << std::endl;
    exit(-1);
  }

  printTimeElapsed( timer ); 

  // Print scene stats
  {
    std::cerr << "Loaded scene: " << config.scene_filename << std::endl;
    std::cerr << "\t" << scene.num_meshes << " meshes, " << scene.num_instances << " instances" << std::endl;
    size_t num_vertices = 0;
    size_t num_triangles = 0;
    for (size_t i = 0; i < scene.num_meshes; ++i) {
      num_vertices += scene.meshes[i].num_vertices;
      num_triangles += scene.meshes[i].num_triangles;
    }
    std::cerr << "\tuninstanced vertices: " << num_vertices << std::endl;
    std::cerr << "\tuninstanced triangles: " << num_triangles << std::endl;
  }

  // OptiX Prime requires all instances to have the same vertex stride
  for (size_t i = 1; i < scene.num_meshes; ++i) {
    if (scene.meshes[i].vertex_stride_bytes != scene.meshes[0].vertex_stride_bytes) {
      std::cerr << "Error: all meshes must have the same vertex stride.  Bailing.\n";
      exit(-1);
    }
  }


  //
  // Generate AO samples
  //

  std::cerr << "Minimum samples per face: " << config.min_samples_per_face << std::endl;

  std::cerr << "Generate sample points ... \n"; std::cerr.flush();

  timer.reset();
  timer.start();
  

  std::vector<size_t> num_samples_per_instance(scene.num_instances);
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
    make_ground_plane(scene_bbox_min, scene_bbox_max, scene.meshes[0].vertex_stride_bytes, 
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
  float** vertex_ao = new float*[ scene.num_instances ];
  for (size_t i = 0; i < scene.num_instances; ++i ) {
    vertex_ao[i] = new float[ scene.meshes[scene.instances[i].mesh_index].num_vertices ];
  }
  bake::mapAOToVertices( scene, &num_samples_per_instance[0], ao_samples, &ao_values[0], config.filter_mode, config.regularization_weight, vertex_ao );

  printTimeElapsed( timer ); 

  //
  // Visualize results
  //
  std::cerr << "Launch viewer  ... \n" << std::endl;
  bake::view( scene.meshes, scene.num_meshes, scene.instances, scene.num_instances, vertex_ao, scene_bbox_min, scene_bbox_max );

  for (size_t i = 0; i < scene.num_instances; ++i) {
    delete [] vertex_ao[i];
  }
  delete [] vertex_ao;

  destroy_ao_samples( ao_samples );

  delete scene_memory;
  
  return 1;
}

// Required logging function
void sample_print(int level, const char * fmt)
{
  //stub
}

