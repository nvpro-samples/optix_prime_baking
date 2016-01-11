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
      else if ( (arg == "-l" || arg == "--least_squares " ) && i+1 < argc ) {
        int flag = 0;
        if( sscanf( argv[++i], "%d", &flag ) != 1 ) {
          printParseErrorAndExit( argv[0], arg, argv[i] );
        }
        if (flag) {
          filter_mode = bake::VERTEX_FILTER_LEAST_SQUARES;
        } else {
          filter_mode = bake::VERTEX_FILTER_AREA_BASED;  
        }
      }
      else if ( (arg == "-w" || arg == "--regularization_weight" ) && i+1 < argc ) {
        if( sscanf( argv[++i], "%f", &regularization_weight ) != 1 ) {
          printParseErrorAndExit( argv[0], arg, argv[i] );
        }
        regularization_weight = std::min( std::max( regularization_weight, 0.0f ), 1.0f );
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
    << "  -l  | --least_squares <0|1>           Enable or disable least squares filtering (default 1 if built with Eigen3)\n"
    << "  -o  | --obj <obj_file>                Specify model to be rendered\n"
    << "  -i  | --instances <n>                 Number of instances per mesh (default 1).  For testing.\n"
    << "  -r  | --rays    <n>                   Number of rays per sample point for gather (default " << NUM_RAYS << ")\n"
    << "  -s  | --samples <n>                   Number of sample points on mesh (default " << SAMPLES_PER_FACE << " per face; any extra samples are based on area)\n"
    << "  -t  | --samples_per_face <n>          Minimum number of samples per face (default " << SAMPLES_PER_FACE << ")\n"
    << "  -w  | --regularization_weight <w>     Regularization weight for least squares, 0-1 range. (default 0.1)\n"
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

  void make_debug_instances(bake::Mesh* bake_mesh, size_t n, std::vector<bake::Instance>& instances)
  {

    // Make up a transform per instance
    const float3 bbox_base = optix::make_float3(0.5f*(bake_mesh->bbox_min[0] + bake_mesh->bbox_max[0]),
                                                      bake_mesh->bbox_min[1],
                                                0.5f*(bake_mesh->bbox_min[2] + bake_mesh->bbox_max[2]));
    const float rot = M_PI/6.0f;
    const float3 rot_axis = optix::make_float3(0.0f, 1.0f, 0.0f);
    const float scale_factor = 0.9f;
    float scale = scale_factor;
    const float3 base_translation = 1.01*optix::make_float3(bake_mesh->bbox_max[0] - bake_mesh->bbox_min[0], 0.0f, 0.0f);
    float3 translation = scale_factor* base_translation;

    for (size_t i = 0; i < n; i++) {
      bake::Instance instance;
      instance.mesh = bake_mesh;
      const optix::Matrix4x4 mat = optix::Matrix4x4::translate(translation) *
                                   optix::Matrix4x4::translate(bbox_base) *
                                   optix::Matrix4x4::rotate((i+1)*rot, rot_axis) *
                                   optix::Matrix4x4::scale(optix::make_float3(scale)) *
                                   optix::Matrix4x4::translate(-bbox_base);
      scale *= scale_factor;
      translation += scale*base_translation;

      const float* matdata = mat.getData();
      std::copy(matdata, matdata+16, instance.xform);
      instances.push_back(instance);
    }
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
  std::cerr << "Load mesh ...              "; std::cerr.flush();

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
  
  std::cerr << "Minimum samples per face: " << config.min_samples_per_face << std::endl;

  //
  // Populate instances
  //
  const size_t num_instances = meshes.size() * config.num_instances_per_mesh;
  std::vector<bake::Instance> instances;
  instances.reserve(num_instances);

  for (size_t meshIdx = 0; meshIdx < meshes.size(); ++meshIdx) {
    
    tinyobj::mesh_t& mesh = meshes[meshIdx];
    bake::Mesh* bake_mesh = new bake::Mesh;
    bake_mesh->num_vertices  = mesh.positions.size();
    bake_mesh->num_normals   = mesh.normals.size(); 
    bake_mesh->num_triangles = mesh.indices.size()/3;
    bake_mesh->vertices      = &mesh.positions[0];
    bake_mesh->normals       = mesh.normals.empty() ? NULL : &mesh.normals[0];
    bake_mesh->tri_vertex_indices = &mesh.indices[0];
    bake_mesh->tri_normal_indices = mesh.normals.empty() ? NULL : &mesh.indices[0];  //Note: tinyobj flattens mesh data

    // Build bbox for mesh

    std::fill(bake_mesh->bbox_min, bake_mesh->bbox_min+3, FLT_MAX);
    std::fill(bake_mesh->bbox_max, bake_mesh->bbox_max+3, -FLT_MAX);
    for (size_t i = 0; i < mesh.positions.size()/3; ++i) {
      for (size_t k = 0; k < 3; ++k) {
        bake_mesh->bbox_min[k] = std::min(bake_mesh->bbox_min[k], mesh.positions[3*i+k]);
        bake_mesh->bbox_max[k] = std::max(bake_mesh->bbox_max[k], mesh.positions[3*i+k]);
      }
    }

    // Make instance

    bake::Instance instance;
    instance.mesh = bake_mesh;  //leak
    const optix::Matrix4x4 mat = optix::Matrix4x4::identity();  // TODO: use transform from file
    const float* matdata = mat.getData();
    std::copy(matdata, matdata+16, instance.xform);
    instances.push_back(instance);

    if (config.num_instances_per_mesh > 1) make_debug_instances(bake_mesh, config.num_instances_per_mesh-1, instances);
    
  }

  assert(instances.size() == num_instances);

  //
  // Generate AO samples
  //

  std::cerr << "Generate sample points ... "; std::cerr.flush();

  timer.reset();
  timer.start();

  std::vector<bake::AOSamples> ao_samples(num_instances);
  const size_t total_samples = bake::sampleSurfaces( &instances[0], instances.size(), config.min_samples_per_face, config.num_samples, &ao_samples[0]);
  
  printTimeElapsed( timer ); 

  std::cerr << "Total samples: " << total_samples << std::endl;

  //
  // Evaluate AO samples 
  //
  std::cerr << "Compute AO ...             "; std::cerr.flush();
  
  timer.reset();
  timer.start();
  float** ao_values = new float*[ num_instances ];
  for (size_t i = 0; i < num_instances; ++i) {
    ao_values[i] = new float[ ao_samples[i].num_samples ];
  }
  bake::computeAO( &instances[0], num_instances, &ao_samples[0], config.num_rays, ao_values );
  printTimeElapsed( timer ); 

  std::cerr << "Map AO to vertices  ...    "; std::cerr.flush();

  timer.reset();
  timer.start();
  float** vertex_ao = new float*[ num_instances ];
  for (size_t i = 0; i < num_instances; ++i ) {
    vertex_ao[i] = new float[ instances[i].mesh->num_vertices ];
    bake::mapAOToVertices( *instances[i].mesh, ao_samples[i], ao_values[i], config.filter_mode, config.regularization_weight, vertex_ao[i] );
  }
  printTimeElapsed( timer ); 

  //
  // Visualize results
  //
  std::cerr << "Launch viewer  ... \n" << std::endl;
  bake::view( &instances[0], instances.size(), vertex_ao );
  
  return 1;
}

// Required logging function
void sample_print(int level, const char * fmt)
{
  //stub
}

