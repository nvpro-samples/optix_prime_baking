
#include <algorithm>
#include <iostream>

#include <main.h>
#include <nv_helpers_gl/WindowInertiaCamera.h>
#include <nv_helpers_gl/GLSLProgram.h>

#include "bake_api.h"

const char* vertex_program = 
"#version 330\n"
"#extension GL_ARB_separate_shader_objects : enable\n"
"uniform mat4 object2world;\n"
"uniform mat4 world2screen;\n"
"layout(location=0) in vec3 P;\n"
"layout(location=1) in float occl;\n"
"out gl_PerVertex {\n"
"    vec4  gl_Position;\n"
"};\n"
"layout(location=0) out vec3 outColor;\n"
"void main() {\n"
"   outColor = vec3(occl, occl, occl);\n"
"   gl_Position = world2screen * object2world * vec4(P, 1.0);\n"
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
  const bake::Instance* m_instances;
  const size_t m_num_instances;
  float const* const* m_vertex_ao;
  std::vector<GLuint> m_vaos;

  const vec3f m_initial_eye;
  const vec3f m_initial_lookat;

public:
  GLSLProgram m_prog;

  MyWindow(const bake::Instance* instances,
           const size_t num_instances,
           float const* const* vertex_ao,
           // Initial camera params
           const vec3f& eye,
           const vec3f& lookat,
           const float fov,
           const float clipnear,
           const float clipfar)
  : WindowInertiaCamera(eye, lookat, lookat, fov, clipnear, clipfar), 
    m_instances(instances),
    m_num_instances(num_instances),
    m_vertex_ao(vertex_ao),
    m_initial_eye(eye),
    m_initial_lookat(lookat),
    m_prog("Mesh Program") {}

  virtual bool init()
  {
    if (!WindowInertiaCamera::init()) return false;

    if (!m_prog.compileProgram(vertex_program, NULL, fragment_program)) return false;

    m_vaos.resize(m_num_instances);
    glGenVertexArrays(m_num_instances, &m_vaos[0]);
    for (size_t i = 0; i < m_num_instances; ++i) {
      glBindVertexArray(m_vaos[i]);

      // Vertex attributes

      GLuint vbos[] = {0, 0};
      glGenBuffers(2, vbos);

      // Positions
      glBindBuffer(GL_ARRAY_BUFFER, vbos[0]);

      const size_t vertex_count = m_instances[i].mesh->num_vertices;
      const float* positions = m_instances[i].mesh->vertices;
      glBufferData(GL_ARRAY_BUFFER, sizeof(float)*vertex_count*3, positions, GL_STATIC_DRAW);
      glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, /*stride*/ 0, /*offset*/ 0);
      glEnableVertexAttribArray(0);

      // Occlusion values
      glBindBuffer(GL_ARRAY_BUFFER, vbos[1]);
      glBufferData(GL_ARRAY_BUFFER, sizeof(float)*vertex_count, m_vertex_ao[i], GL_STATIC_DRAW);
      glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, /*stride*/ 0, /*offset*/ 0);
      glEnableVertexAttribArray(1);

      // Vertex indices

      GLuint ebo;
      glGenBuffers(1, &ebo);
      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
      const size_t triangle_count = m_instances[i].mesh->num_triangles;
      const unsigned int* indices = m_instances[i].mesh->tri_vertex_indices; 
      glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int)*triangle_count*3, indices, GL_STATIC_DRAW);
    }

    glEnable(GL_DEPTH_TEST);

    return true;
  }

  virtual void display()
  {
    WindowInertiaCamera::display();

    mat4f world2screen = m_projection * m_camera.m4_view;
    m_prog.setUniformMatrix4fv("world2screen", world2screen.mat_array, /*transpose*/ false);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    m_prog.enable();
    for (size_t i = 0; i < m_num_instances; ++i) {
      
      glBindVertexArray(m_vaos[i]);

      // Note: optix matrix is transposed from opengl
      m_prog.setUniformMatrix4fv("object2world", const_cast<GLfloat*>(m_instances[i].xform), /*transpose*/ true);

      const size_t num_triangles = m_instances[i].mesh->num_triangles;
      const GLsizei num_indices = static_cast<GLsizei>(num_triangles*3);
      glDrawElements(GL_TRIANGLES, num_indices, GL_UNSIGNED_INT, 0);
    }

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
      case 'f':  // frame the scene
      case 'F':
      {
        const vec3f current_view_vector = normalize(m_camera.curFocusPos - m_camera.curEyePos);
        const float initial_view_dist = length(m_initial_lookat - m_initial_eye);
        m_camera.look_at(vec3f(m_initial_lookat - initial_view_dist*current_view_vector), m_initial_lookat, /*reset*/ true);
        break;
      }

    }
  }
  
}; // MyWindow



namespace bake {

  void view( const bake::Instance* instances, const size_t num_instances, float const* const* vertex_colors )
  {

    // bbox for all instances
    vec3f bbox_min(FLT_MAX, FLT_MAX, FLT_MAX);
    vec3f bbox_max(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    for (size_t i = 0; i < num_instances; ++i) {
      mat4f object2world(instances[i].xform);
      object2world = transpose(object2world);  // optix matrixes are transposed from opengl
      vec3f instance_bbox_min = object2world*vec3f(instances[i].mesh->bbox_min);
      vec3f instance_bbox_max = object2world*vec3f(instances[i].mesh->bbox_max);
      for(size_t k = 0; k < 3; ++k) {
        bbox_min[k] = std::min(bbox_min[k], instance_bbox_min[k]);
        bbox_max[k] = std::max(bbox_max[k], instance_bbox_max[k]);
      }
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

    static MyWindow window(instances, num_instances, vertex_colors, eye, lookat, fov, clipnear, clipfar);

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



