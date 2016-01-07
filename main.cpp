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
  std::string obj_filename;
  int num_samples;
  int min_samples_per_face;
  int num_rays;
  bake::VertexFilterMode filter_mode;
  float regularization_weight;

  Config( int argc, const char ** argv ) {
    // set defaults
#ifdef PROJECT_ABSDIRECTORY
    obj_filename = std::string(PROJECT_ABSDIRECTORY) + std::string("/assets/lucy.obj");
#else
    obj_filename = std::string( "./assets/lucy.obj" );
#endif
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
        obj_filename = argv[++i];
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
  }

  void printUsageAndExit( const char* argv0 )
  {
    std::cerr
    << "Usage  : " << argv0 << " [options]\n"
    << "App options:\n"
    << "  -h  | --help                          Print this usage message\n"
    << "  -l  | --least_squares <0|1>           Enable or disable least squares filtering (default 1 if built with Eigen3)\n"
    << "  -o  | --obj <obj_file>                Specify model to be rendered\n"
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
  tinyobj::mesh_t mesh;
  { 
    std::string errs;
    bool loaded = tinyobj::LoadObj(mesh, errs, config.obj_filename.c_str());
    if (!errs.empty() || !loaded) {
      std::cerr << errs << std::endl;
      return 0; 
    }
  }

  printTimeElapsed( timer ); 
  
  std::cerr << "Minimum samples per face: " << config.min_samples_per_face << std::endl;

  //
  // Populate instances
  //
  const size_t num_user_meshes = 1;
  const size_t num_instances_per_mesh = 1;
  const size_t num_instances = num_user_meshes * num_instances_per_mesh;
  std::vector<bake::Instance> instances;
  instances.reserve(num_instances);

  for (size_t meshIdx = 0; meshIdx < num_user_meshes; ++meshIdx) {
    
    bake::Mesh* bake_mesh = new bake::Mesh;
    bake_mesh->num_vertices  = mesh.positions.size();
    bake_mesh->num_normals   = mesh.normals.size(); 
    bake_mesh->num_triangles = mesh.indices.size()/3;
    bake_mesh->vertices      = &mesh.positions[0];
    bake_mesh->normals       = mesh.normals.empty() ? NULL : &mesh.normals[0];
    bake_mesh->tri_vertex_indices = &mesh.indices[0];
    bake_mesh->tri_normal_indices = mesh.normals.empty() ? NULL : &mesh.indices[0];  //Note: tinyobj flattens mesh data

    // Get bbox

    std::fill(bake_mesh->bbox_min, bake_mesh->bbox_min+3, FLT_MAX);
    std::fill(bake_mesh->bbox_max, bake_mesh->bbox_max+3, -FLT_MAX);

    for (size_t i = 0; i < mesh.positions.size()/3; ++i) {
      bake_mesh->bbox_min[0] = std::min(bake_mesh->bbox_min[0], mesh.positions[3*i]);
      bake_mesh->bbox_max[0] = std::max(bake_mesh->bbox_max[0], mesh.positions[3*i]);
      bake_mesh->bbox_min[1] = std::min(bake_mesh->bbox_min[1], mesh.positions[3*i+1]);
      bake_mesh->bbox_max[1] = std::max(bake_mesh->bbox_max[1], mesh.positions[3*i+1]);
      bake_mesh->bbox_min[2] = std::min(bake_mesh->bbox_min[2], mesh.positions[3*i+2]);
      bake_mesh->bbox_max[2] = std::max(bake_mesh->bbox_max[2], mesh.positions[3*i+2]);
    }

    bake::Instance instance;
    instance.mesh = bake_mesh; // leak
    optix::Matrix4x4 mat = optix::Matrix4x4::identity();
    const float* matdata = mat.getData();
    std::copy(matdata, matdata+16, instance.xform);

    instances.push_back(instance);
  }

  //
  // Generate AO samples
  //

  std::cerr << "Generate sample points ... "; std::cerr.flush();

  timer.reset();
  timer.start();

  std::vector<bake::AOSamples> ao_samples(num_instances);
  bake::sampleSurfaces( &instances[0], instances.size(), config.min_samples_per_face, config.num_samples, &ao_samples[0]);
  
  printTimeElapsed( timer ); 

  {
    size_t total_samples = 0;
    for (size_t i = 0; i < num_instances; ++i) {
      total_samples += ao_samples[i].num_samples;
    }
    std::cerr << "Total samples: " << total_samples << std::endl;
  }

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
  bake::view( *instances[0].mesh, vertex_ao[0] );  // TODO
  
  return 1;
}

// Required logging function
void sample_print(int level, const char * fmt)
{
  //stub
}

