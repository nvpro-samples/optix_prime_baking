
#include <algorithm>
#include <iostream>

#include <main.h>
#include <nv_helpers_gl/WindowInertiaCamera.h>
#include <nv_helpers_gl/GLSLProgram.h>

#include "bake_api.h"

const char* vertex_program = 
"#version 330\n"
"#extension GL_ARB_separate_shader_objects : enable\n"
"uniform mat4 xform;\n"
"layout(location=0) in vec3 P;\n"
"layout(location=1) in float occl;\n"
"out gl_PerVertex {\n"
"    vec4  gl_Position;\n"
"};\n"
"layout(location=0) out vec3 outColor;\n"
"void main() {\n"
"   outColor = vec3(occl, occl, occl);\n"
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
  const float* m_positions;
  const float* m_vertex_ao;
  const size_t m_vertex_count;
  const unsigned int* m_indices;
  const size_t m_triangle_count;

public:
  GLSLProgram m_prog;

  MyWindow(const float* positions, 
           const float* vertex_ao,
           const size_t vertex_count,
           const unsigned int* indices,
           const size_t triangle_count,
           // Initial camera params
           const vec3f& eye,
           const vec3f& lookat,
           const float fov,
           const float clipnear,
           const float clipfar)
  : WindowInertiaCamera(eye, lookat, lookat, fov, clipnear, clipfar), 
    m_positions(positions),
    m_vertex_ao(vertex_ao),
    m_vertex_count(vertex_count),
    m_indices(indices),
    m_triangle_count(triangle_count),
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
    glBindBuffer(GL_ARRAY_BUFFER, vbos[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float)*m_vertex_count*3, m_positions, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, /*stride*/ 0, /*offset*/ 0);
    glEnableVertexAttribArray(0);

    // Occlusion values
    glBindBuffer(GL_ARRAY_BUFFER, vbos[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float)*m_vertex_count, m_vertex_ao, GL_STATIC_DRAW);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, /*stride*/ 0, /*offset*/ 0);
    glEnableVertexAttribArray(1);

    // Vertex indices

    GLuint ebo;
    glGenBuffers(1, &ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int)*m_triangle_count*3, m_indices, GL_STATIC_DRAW);

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

    const GLsizei num_indices = static_cast<GLsizei>(m_triangle_count*3);
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



namespace bake {

  void view( const bake::Mesh& bake_mesh, const float* vertex_colors )
  {

    vec3f bbox_min(bake_mesh.bbox_min[0], bake_mesh.bbox_min[1], bake_mesh.bbox_min[2]);
    vec3f bbox_max(bake_mesh.bbox_max[0], bake_mesh.bbox_max[1], bake_mesh.bbox_max[2]);
    const vec3f center = 0.5f*(bbox_min + bbox_max);
    const vec3f bbox_extents = bbox_max - bbox_min;
    const float max_extent = std::max( std::max(bbox_extents[0], bbox_extents[1]), bbox_extents[2]);

    // Initial camera params
    const vec3f eye = center + vec3f(0, 0, 2.0f*max_extent);
    const vec3f lookat = center;
    const float fov = 30.0f;
    const float clipnear = 0.01f*max_extent;
    const float clipfar = 10.0f*max_extent;

    static MyWindow window(bake_mesh.vertices, vertex_colors, bake_mesh.num_vertices, bake_mesh.tri_vertex_indices, bake_mesh.num_triangles,
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

    if (!window.create("Baked AO Viewer", &context)) return;

    window.makeContextCurrent();
    window.swapInterval(0);

    while(MyWindow::sysPollEvents(false)) {
      window.idle();
    }

  }

} //namespace



