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
const float  GROUND_SCALE = 100.0f;
const float  GROUND_OFFSET = 0.03f;
const float  SCENE_OFFSET_SCALE = 0.01f;
const float  SCENE_MAXDISTANCE_SCALE = 1.1f;
const float  REGULARIZATION_WEIGHT = 0.1f;
const char* DEFAULT_BK3DGZ_FILE = "sled_v134.bk3d.gz";
const char* DEFAULT_BK3D_FILE = "lucy_v134.bk3d";
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
  bool use_viewer;
  int  ground_upaxis;
  float ground_scale_factor;
  float ground_offset_factor;
  float scene_offset_scale;
  float scene_maxdistance_scale;
  float scene_maxdistance;
  float scene_offset;
  bool  use_cpu;
  bool  conserve_memory;
  bool  flip_orientation;
  std::string output_filename;

  Config( int argc, const char ** argv ) {
    // set defaults
    num_instances_per_mesh = 1;
    num_samples = 0;  // default means determine from mesh
    min_samples_per_face = SAMPLES_PER_FACE;
    num_rays    = NUM_RAYS; 
    ground_upaxis = 1;
    ground_scale_factor  = GROUND_SCALE;
    ground_offset_factor = GROUND_OFFSET;
    scene_offset_scale = SCENE_OFFSET_SCALE;
    scene_maxdistance_scale = SCENE_MAXDISTANCE_SCALE;
    scene_offset = 0; // must default to 0
    scene_maxdistance = 0;
    use_cpu = false;
    conserve_memory = false;
    flip_orientation = false;
#ifdef EIGEN3_ENABLED
    filter_mode = bake::VERTEX_FILTER_LEAST_SQUARES;
#else
    filter_mode = bake::VERTEX_FILTER_AREA_BASED;
#endif
    regularization_weight = REGULARIZATION_WEIGHT;
    use_ground_plane_blocker = true;
    use_viewer = true;


    // parse arguments
    for ( int i = 1; i < argc; ++i ) 
    { 
      std::string arg( argv[i] );
      if ( arg.empty() ) continue;

      if( arg == "-h" || arg == "--help" ) 
      {
        printUsageAndExit( argv[0] ); 
      } 
      else if( (arg == "-f" || arg == "--file") && i+1 < argc ) 
      {
        assert( scene_filename.empty() && "multiple -f (--file) flags found when parsing command line");
        scene_filename = argv[++i];
      } 
      else if ((arg == "-o" || arg == "--outfile") && i + 1 < argc)
      {
        assert(output_filename.empty() && "multiple -o (--outfile) flags found when parsing command line");
        output_filename = argv[++i];
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
      else if ((arg == "-d" || arg == "--ray_distance_scale") && i + 1 < argc)
      {
        if (sscanf(argv[++i], "%f", &scene_offset_scale) != 1) {
          printParseErrorAndExit(argv[0], arg, argv[i]);
        }
      }
      else if ((arg == "-m" || arg == "--hit_distance_scale") && i + 1 < argc)
      {
        if (sscanf(argv[++i], "%f", &scene_maxdistance_scale) != 1) {
          printParseErrorAndExit(argv[0], arg, argv[i]);
        }
      }
      else if ((arg == "--ray_distance") && i + 1 < argc)
      {
        if (sscanf(argv[++i], "%f", &scene_offset) != 1) {
          printParseErrorAndExit(argv[0], arg, argv[i]);
        }
      }
      else if ((arg == "--hit_distance") && i + 1 < argc)
      {
        if (sscanf(argv[++i], "%f", &scene_maxdistance) != 1) {
          printParseErrorAndExit(argv[0], arg, argv[i]);
        }
      }
      else if ( (arg == "-r" || arg == "--rays") && i+1 < argc )
      {
        if( sscanf( argv[++i], "%d", &num_rays ) != 1 ) {
          printParseErrorAndExit( argv[0], arg, argv[i] );
        }
      }
      else if ((arg == "-g" || arg == "--ground_setup") && i + 3 < argc)
      {
        if (sscanf(argv[++i], "%d", &ground_upaxis) != 1 || (ground_upaxis < 0 || ground_upaxis > 5)) {
          printParseErrorAndExit(argv[0], arg, argv[i]);
        }
        if (sscanf(argv[++i], "%f", &ground_scale_factor) != 1) {
          printParseErrorAndExit(argv[0], arg, argv[i]);
        }
        if (sscanf(argv[++i], "%f", &ground_offset_factor) != 1) {
          printParseErrorAndExit(argv[0], arg, argv[i]);
        }
      }
      else if ((arg == "--flip_orientation")) {
        flip_orientation = true;
      }
      else if ( (arg == "--no_ground_plane" ) ) {
        use_ground_plane_blocker = false;
      }
      else if ((arg == "--no_viewer")) {
        use_viewer = false;
      }
      else if ((arg == "--no_gpu")) {
        use_cpu = true;
      }
      else if ((arg == "--conserve_memory")) {
        conserve_memory = true;
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
      // Try more interesting gzipped file first
      std::string bk3d_path = asset_path + std::string(DEFAULT_BK3DGZ_FILE);
      struct stat buf;
      if (stat(bk3d_path.c_str(), &buf) == 0) {
        scene_filename = bk3d_path;
      }
#endif
      if (scene_filename.empty()) {
        // Fall back to simpler file
        std::string obj_path = asset_path + std::string(DEFAULT_BK3D_FILE);
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
    << "  -f  | --file <scene_file>             Specify model to be rendered (obj, bk3d, bk3d.gz, csf, csf.gz).\n"
    << "  -o  | --outfile <vertex_ao_file>      Specify raw file where per-instance ao vertices are stored (very basic fileformat).\n"
    << "  -i  | --instances <n>                 Number of instances per mesh (default 1).  For testing.\n"
    << "  -r  | --rays    <n>                   Number of rays per sample point for gather (default " << NUM_RAYS << ")\n"
    << "  -s  | --samples <n>                   Number of sample points on mesh (default " << SAMPLES_PER_FACE << " per face; any extra samples are based on area)\n"
    << "  -t  | --samples_per_face <n>          Minimum number of samples per face (default " << SAMPLES_PER_FACE << ")\n"
    << "  -d  | --ray_distance_scale <s>        Distance offset scale for ray from face: ray offset = maximum scene extent * s. (default " << SCENE_OFFSET_SCALE << ")\n"
    << "        --ray_distance <s>              Distance offset scale for ray from face: ray offset = s. (overrides scale-based version, used if non zero)\n"
    << "  -m  | --hit_distance_scale <s>        Maximum hit distance to contribute: max distance = maximum scene extent * s. (default " << SCENE_MAXDISTANCE_SCALE << ")\n"
    << "        --hit_distance <s>              Maximum hit distance to contribute: max distance = s. (overrides scale-based version, used if non zero)\n"
    << "  -g  | --ground_setup <axis> <s> <o>   Ground plane setup: axis(int 0,1,2,3,4,5 = +x,+y,+z,-x,-y,-z) scale(float) offset(float). "
    <<                                          " (default 1 " << GROUND_SCALE << " " << GROUND_OFFSET << ")\n"
    << "        --flip_orientation              Flips model winding and vertex normals (useful for storing two-sided baking results separately)\n"
    << "        --no_ground_plane               Disable virtual ground plane\n"
    << "        --no_viewer                     Disable OpenGL viewer\n"
    << "        --no_gpu                        Disable GPU usage in raytracer\n"
    << "        --conserve_memory               Triggers some internal settings in optix to save memory\n"
#ifdef EIGEN3_ENABLED
    << "  -w  | --regularization_weight <w>     Regularization weight for least squares, positive range. (default " << REGULARIZATION_WEIGHT << ")\n"
    << "        --no_least_squares              Disable least squares filtering\n"
#endif
    << std::endl
    << "Viewer keys:\n"
    << "   e                                    Draw mesh edges on/off\n"
    << "   f                                    Frame scene\n"
    << "   q                                    Quit\n"
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

  void set_vertex_entry(float* vertices, int idx, int axis, float* vec)
  {
    vertices[3 * idx + axis] = vec[axis];
  }

  void make_ground_plane(float scene_bbox_min[3], float scene_bbox_max[3],
                         unsigned upaxis, float scale_factor, float offset_factor,
                         unsigned scene_vertex_stride_bytes,
                         std::vector<float>& plane_vertices, std::vector<unsigned int>& plane_indices,
                         std::vector<bake::Mesh>& meshes, std::vector<bake::Instance>& instances)
  {

    const unsigned int index_data[] = {0, 1, 2, 0, 2, 3, 2, 1, 0, 3, 2, 0};
    unsigned int num_indices = sizeof(index_data) / sizeof(index_data[0]);
    plane_indices.resize(num_indices);
    std::copy(index_data, index_data + num_indices, plane_indices.begin());
    float scene_extents[] = {scene_bbox_max[0] - scene_bbox_min[0],
                             scene_bbox_max[1] - scene_bbox_min[1],
                             scene_bbox_max[2] - scene_bbox_min[2]};
    
    float ground_min[] = {scene_bbox_max[0] - scale_factor*scene_extents[0],
                          scene_bbox_min[1] - scale_factor*scene_extents[1],
                          scene_bbox_max[2] - scale_factor*scene_extents[2]};
    float ground_max[] = {scene_bbox_min[0] + scale_factor*scene_extents[0],
                          scene_bbox_min[1] + scale_factor*scene_extents[1],
                          scene_bbox_min[2] + scale_factor*scene_extents[2]};

    if (upaxis > 2){
      upaxis %= 3;
      ground_min[upaxis] = scene_bbox_max[upaxis] + scene_extents[upaxis] * offset_factor;
      ground_max[upaxis] = scene_bbox_max[upaxis] + scene_extents[upaxis] * offset_factor;
    }
    else{
      ground_min[upaxis] = scene_bbox_min[upaxis] - scene_extents[upaxis] * offset_factor;
      ground_max[upaxis] = scene_bbox_min[upaxis] - scene_extents[upaxis] * offset_factor;
    }

    int axis0 = (upaxis + 2) % 3;
    int axis1 = (upaxis + 1) % 3;

    float vertex_data[4 * 3] = {};
    set_vertex_entry(vertex_data, 0, upaxis, ground_min);
    set_vertex_entry(vertex_data, 0, axis0, ground_min);
    set_vertex_entry(vertex_data, 0, axis1, ground_min);
    
    set_vertex_entry(vertex_data, 1, upaxis, ground_min);
    set_vertex_entry(vertex_data, 1, axis0, ground_max);
    set_vertex_entry(vertex_data, 1, axis1, ground_min);

    set_vertex_entry(vertex_data, 2, upaxis, ground_min);
    set_vertex_entry(vertex_data, 2, axis0, ground_max);
    set_vertex_entry(vertex_data, 2, axis1, ground_max);

    set_vertex_entry(vertex_data, 3, upaxis, ground_min);
    set_vertex_entry(vertex_data, 3, axis0, ground_min);
    set_vertex_entry(vertex_data, 3, axis1, ground_max);
    


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
    plane_mesh.num_triangles = num_indices/3;
    plane_mesh.vertices      = &plane_vertices[0];
    plane_mesh.vertex_stride_bytes = vertex_stride_bytes;
    plane_mesh.normals       = NULL;
    plane_mesh.normal_stride_bytes = 0;
    plane_mesh.tri_vertex_indices = &plane_indices[0];
    
    bake::Instance instance;
    instance.mesh_index = (unsigned int)meshes.size();

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

  void allocate_ao_samples(bake::AOSamples& ao_samples, size_t n) {
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

  bool save_results(const char* outputfile, bake::Scene & scene, const float* const * ao_vertex)
  {
    FILE* file = fopen(outputfile, "wb");
    if (!file) return false;

    uint64_t numInstances = scene.num_instances;
    uint64_t numVertices = 0;
    
    for (size_t i = 0; i < scene.num_instances; i++){
      numVertices += scene.meshes[scene.instances[i].mesh_index].num_vertices;
    }

    // write header
    fwrite(&numInstances, sizeof(numInstances), 1, file);
    fwrite(&numVertices, sizeof(numVertices), 1, file);
    
    // write instances

    uint64_t vertexOffset = 0;
    for (size_t i = 0; i < scene.num_instances; i++){
      uint64_t identifier = scene.instances[i].storage_identifier;
      fwrite(&identifier, sizeof(identifier), 1, file);
      fwrite(&vertexOffset, sizeof(vertexOffset), 1, file);
      numVertices = scene.meshes[scene.instances[i].mesh_index].num_vertices;
      fwrite(&numVertices, sizeof(numVertices), 1, file);

      vertexOffset += numVertices;
    }

    // write vertices
    for (size_t i = 0; i < scene.num_instances; i++){
      numVertices = scene.meshes[scene.instances[i].mesh_index].num_vertices;
      fwrite(ao_vertex[i], sizeof(float)*numVertices, 1, file);
    }
    
    fflush(file);
    fclose(file);

    return true;
  }

  // Concat two scenes using shallow copies for all buffers
  void concat_scenes( const bake::Scene& scene1, const bake::Scene& scene2, 
    //output
    bake::Scene& scene, std::vector<bake::Mesh>& meshes, std::vector<bake::Instance>& instances )
  {
    // Concat array of meshes
    for (size_t i = 0; i < scene1.num_meshes; ++i) meshes.push_back( scene1.meshes[i] );
    for (size_t i = 0; i < scene2.num_meshes; ++i) meshes.push_back( scene2.meshes[i] );
    scene.num_meshes = scene1.num_meshes + scene2.num_meshes;
    scene.meshes = &meshes[0];

    // Concat array of instances, with updated mesh offsets
    for (size_t i = 0; i < scene1.num_instances; ++i) instances.push_back( scene1.instances[i] );
    for (size_t i = 0; i < scene2.num_instances; ++i) {
      bake::Instance instance = scene2.instances[i];
      instance.mesh_index += scene1.num_meshes;
      instances.push_back( instance );
    }
    scene.num_instances = scene1.num_instances + scene2.num_instances;
    scene.instances = &instances[0];
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

  if (config.flip_orientation){
    for (size_t m = 0; m < scene.num_meshes; ++m) {
      bake::Mesh& mesh = scene.meshes[m];
      for (size_t i = 0; i < mesh.num_triangles; i++){
        std::swap(mesh.tri_vertex_indices[i * 3 + 0], mesh.tri_vertex_indices[i * 3 + 2]);
      }
      if (mesh.normals){
        size_t stride = mesh.normal_stride_bytes / sizeof(float);
        for (size_t i = 0; i < mesh.num_vertices; i++){
          mesh.normals[i * stride + 0] *= -1.0;
          mesh.normals[i * stride + 1] *= -1.0;
          mesh.normals[i * stride + 2] *= -1.0;
        }
      }
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
  {
    const int sqrt_num_rays = static_cast<int>( sqrtf( static_cast<float>( config.num_rays ) ) + .5f );
    std::cerr << "Rays per sample: " << sqrt_num_rays * sqrt_num_rays << std::endl;
    std::cerr << "Total rays: " << total_samples * sqrt_num_rays * sqrt_num_rays << std::endl;
  }

  //
  // Evaluate AO samples 
  //
  std::cerr << "Compute AO ...             "; std::cerr.flush();
  
  timer.reset();
  timer.start();

  std::vector<float> ao_values( total_samples );
  std::fill(ao_values.begin(), ao_values.end(), 0.0f);


  float scene_maxdistance;
  float scene_offset;
  {
    const float scene_scale = std::max(std::max(scene_bbox_max[0] - scene_bbox_min[0],
                                                scene_bbox_max[1] - scene_bbox_min[1]),
                                                scene_bbox_max[2] - scene_bbox_min[2]);
    scene_maxdistance = scene_scale * config.scene_maxdistance_scale;
    scene_offset = scene_scale * config.scene_offset_scale;
    if (config.scene_offset){
      scene_offset = config.scene_offset;
    }
    if (config.scene_maxdistance){
      scene_maxdistance = config.scene_maxdistance;
    }
  }

  if (config.use_ground_plane_blocker) {
    // Add blocker for ground plane (no surface samples)
    std::vector<bake::Mesh> blocker_meshes;
    std::vector<bake::Instance> blocker_instances;
    std::vector<float> plane_vertices;
    std::vector<unsigned int> plane_indices;
    make_ground_plane(scene_bbox_min, scene_bbox_max, config.ground_upaxis, config.ground_scale_factor, config.ground_offset_factor,
      scene.meshes[0].vertex_stride_bytes, 
      plane_vertices, plane_indices, blocker_meshes, blocker_instances);
    bake::Scene blockers = { &blocker_meshes[0], blocker_meshes.size(), &blocker_instances[0], blocker_instances.size() };

    bake::Scene combined_scene;
    std::vector<bake::Mesh> combined_meshes;
    std::vector<bake::Instance> combined_instances;
    concat_scenes( scene, blockers, combined_scene, combined_meshes, combined_instances );
    bake::computeAO(combined_scene,
      ao_samples, config.num_rays, scene_offset, scene_maxdistance, config.use_cpu, config.conserve_memory, &ao_values[0]);
  } else {
    bake::computeAO(scene, 
      ao_samples, config.num_rays, scene_offset, scene_maxdistance, config.use_cpu, config.conserve_memory, &ao_values[0]);
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

  if (!config.output_filename.empty())
  {
    std::cerr << "Save vertex ao ...              "; std::cerr.flush();
    timer.reset();
    timer.start();
    bool saved = save_results(config.output_filename.c_str(), scene, vertex_ao);
    printTimeElapsed(timer);
    if (saved){
      std::cerr << "Saved vertex ao to: " << config.output_filename << std::endl;
    }
    else{
      std::cerr << "Failed to save vertex ao to: " << config.output_filename << std::endl;
    }    
  }

  if (config.use_viewer){
    //
    // Visualize results
    //
    std::cerr << "Launch viewer  ... \n" << std::endl;
    bake::view(scene.meshes, scene.num_meshes, scene.instances, scene.num_instances, vertex_ao, scene_bbox_min, scene_bbox_max);
  }

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

