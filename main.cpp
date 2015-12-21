
#include <algorithm>
#include <iostream>

#include <main.h>
#include <nv_helpers_gl/WindowInertiaCamera.h>
#include <nv_helpers_gl/GLSLProgram.h>

const char* vertex_program = 
"#version 330\n"
"#extension GL_ARB_separate_shader_objects : enable\n"
"uniform mat4 xform;\n"
"layout(location=0) in vec3 P;\n"
"layout(location=1) in vec3 color;\n"
"out gl_PerVertex {\n"
"    vec4  gl_Position;\n"
"};\n"
"layout(location=0) out vec3 outColor;\n"
"void main() {\n"
"   outColor = color;\n"
"   gl_Position = xform * vec4(P, 1.0);\n"
"   //gl_Position = vec4(P, 1.0);\n"
"}\n"
;

const char* fragment_program =
"#version 330\n"
"#extension GL_ARB_separate_shader_objects : enable\n"
"layout(location=0) in vec3 color;\n"
"layout(location=0) out vec4 outColor;\n"
"void main() {\n"
"   // clamp after vertex interpolation\n"
"   float r = max(min(color.x, 1.0f), 0.0f);\n"
"   float g = max(min(color.y, 1.0f), 0.0f);\n"
"   float b = max(min(color.z, 1.0f), 0.0f);\n"
"   outColor = vec4(r, g, b, 1);\n"
"}\n"
;


class MyWindow: public WindowInertiaCamera
{
private:
  const std::vector<float>& m_positions;
  const std::vector<float>& m_colors;
  const std::vector<unsigned int>& m_indices;

public:
  GLSLProgram m_prog;

  MyWindow(const std::vector<float>& positions, 
           const std::vector<float>& colors,
           const std::vector<unsigned int>& indices,
           // Initial camera params
           const vec3f& eye,
           const vec3f& lookat,
           const float fov,
           const float clipnear,
           const float clipfar)
  : WindowInertiaCamera(eye, lookat, lookat, fov, clipnear, clipfar), 
    m_positions(positions),
    m_colors(colors),
    m_indices(indices),
    m_prog("Mesh Program") {}

  virtual bool init()
  {
    if (!WindowInertiaCamera::init()) return false;

    if (!m_prog.compileProgram(vertex_program, NULL, fragment_program)) return false;

    GLuint vao = 0;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    // Vertex attributes

    GLuint vbos[] = {0, 0};
    glGenBuffers(2, vbos);

    // Positions
    const float* vertex_positions = &m_positions[0];
    glBindBuffer(GL_ARRAY_BUFFER, vbos[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float)*m_positions.size(), vertex_positions, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, /*stride*/ 0, /*offset*/ 0);
    glEnableVertexAttribArray(0);

    // Colors
    const float* vertex_colors = &m_colors[0];
    glBindBuffer(GL_ARRAY_BUFFER, vbos[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float)*m_colors.size(), vertex_colors, GL_STATIC_DRAW);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, /*stride*/ 0, /*offset*/ 0);
    glEnableVertexAttribArray(1);

    // Vertex indices

    GLuint ebo;
    glGenBuffers(1, &ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    const unsigned int* elements = &m_indices[0];
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int)*m_indices.size(), elements, GL_STATIC_DRAW);

    glEnable(GL_DEPTH_TEST);

    return true;
  }

  virtual void display()
  {
    WindowInertiaCamera::display();

    mat4f world2screen = m_projection * m_camera.m4_view;

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    m_prog.enable();
    m_prog.setUniformMatrix4fv("xform", world2screen.mat_array, false);

    const size_t num_indices = m_indices.size();
    glDrawElements(GL_TRIANGLES, num_indices, GL_UNSIGNED_INT, 0);

    swapBuffers();
  }

  virtual void keyboardchar( unsigned char key, int mods, int x, int y )
  {
    switch(key)
    {
      case 'q':
      case 'Q':
        postQuit();
        break;
    }
  }
  
}; // MyWindow

void printUsageAndExit( const char* prog )
{
  std::cerr << prog << " mesh.obj" << std::endl;
  exit( 1 );
}

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"



// Required entry point
int sample_main(int argc, const char** argv)
{

  if (argc < 2) {
    printUsageAndExit( argv[0] );
  }
  tinyobj::mesh_t mesh;
  { 
    std::string errs;
    bool loaded = tinyobj::LoadObj(mesh, errs, argv[1]);
    if (!errs.empty() || !loaded) {
      std::cerr << errs << std::endl;
      return 0; 
    }
  }

  //debug, map normals to colors
  std::vector<float> colors(mesh.positions.size());
  if (mesh.normals.empty()) {
    // Fill with red debug color
    std::fill(colors.begin(), colors.end(), 0.0f);
    for (size_t i = 0; i < colors.size(); i+=3) {
      colors[i] = 1.0f;
    }
  } else {
    for (size_t i = 0; i < mesh.normals.size(); ++i) {
      colors[i] = 0.5f*mesh.normals[i] + 0.5f;
    }
  }

  // Get bbox
  vec3f bbox_min(FLT_MAX, FLT_MAX, FLT_MAX);
  vec3f bbox_max(-FLT_MAX, -FLT_MAX, -FLT_MAX);

  for (size_t i = 0; i < mesh.positions.size()/3; ++i) {
    bbox_min[0] = std::min(bbox_min[0], mesh.positions[3*i]);
    bbox_max[0] = std::max(bbox_max[0], mesh.positions[3*i]);
    bbox_min[1] = std::min(bbox_min[1], mesh.positions[3*i+1]);
    bbox_max[1] = std::max(bbox_max[1], mesh.positions[3*i+1]);
    bbox_min[2] = std::min(bbox_min[2], mesh.positions[3*i+2]);
    bbox_max[2] = std::max(bbox_max[2], mesh.positions[3*i+2]);
  }
  const vec3f center = 0.5f*(bbox_min + bbox_max);
  const vec3f bbox_extents = bbox_max - bbox_min;
  const float max_extent = std::max( std::max(bbox_extents[0], bbox_extents[1]), bbox_extents[2]);

  // Initial camera params
  const vec3f eye = center + vec3f(0, 0, 2.0f*max_extent);
  const vec3f lookat = center;
  const float fov = 30.0f;
  const float clipnear = 0.01f*max_extent;
  const float clipfar = 10.0f*max_extent;
  MyWindow window(mesh.positions, colors, mesh.indices,
    eye, lookat, fov, clipnear, clipfar);
  
  NVPWindow::ContextFlags context(
      1,      //major;
      0,      //minor;
      false,  //core;
      8,      //MSAA;
      24,     //depth bits
      8,      //stencil bits
      true,   //debug;
      false,  //robust;
      false,  //forward;
      NULL    //share;
      );

  if (!window.create("Baked AO Viewer", &context)) return false;

  window.makeContextCurrent();
  window.swapInterval(0);

  while(MyWindow::sysPollEvents(false)) {
    window.idle();
  }

  return 1;
}

// Required logging function
void sample_print(int level, const char * fmt)
{
  //stub
}

