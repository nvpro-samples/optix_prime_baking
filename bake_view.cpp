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
  const bake::Mesh* m_meshes;
  const size_t m_num_meshes;
  const bake::Instance* m_instances;
  const size_t m_num_instances;
  float const* const* m_vertex_ao;
  std::vector<GLuint> m_vaos;

  const vec3f m_initial_eye;
  const vec3f m_initial_lookat;

  bool m_draw_edges;

public:
  GLSLProgram m_prog;

  MyWindow(const bake::Mesh* meshes,
           const size_t num_meshes,
           const bake::Instance* instances,
           const size_t num_instances,
           float const* const* vertex_ao,
           // Initial camera params
           const vec3f& eye,
           const vec3f& lookat,
           const float fov,
           const float clipnear,
           const float clipfar)
  : WindowInertiaCamera(eye, lookat, lookat, fov, clipnear, clipfar), 
    m_meshes(meshes),
    m_num_meshes(num_meshes),
    m_instances(instances),
    m_num_instances(num_instances),
    m_vertex_ao(vertex_ao),
    m_initial_eye(eye),
    m_initial_lookat(lookat),
    m_draw_edges(false),
    m_prog("Mesh Program") {}

  virtual bool init()
  {
    if (!WindowInertiaCamera::init()) return false;

    if (!m_prog.compileProgram(vertex_program, NULL, fragment_program)) return false;

    // Per mesh data (shared by one or more instances)
    std::vector<GLuint> mesh_vbos(m_num_meshes);
    glGenBuffers((GLsizei)m_num_meshes, &mesh_vbos[0]);
    std::vector<GLuint> mesh_ebos(m_num_meshes);
    glGenBuffers((GLsizei)m_num_meshes, &mesh_ebos[0]);

    for (size_t meshIdx = 0; meshIdx < m_num_meshes; ++meshIdx) {

      // Fill position buffer

      glBindBuffer(GL_ARRAY_BUFFER, mesh_vbos[meshIdx]);

      const bake::Mesh& mesh = m_meshes[meshIdx];
      const size_t vertex_count = mesh.num_vertices;
      const float* positions = mesh.vertices;
      const unsigned vertex_stride_bytes = mesh.vertex_stride_bytes > 0 ? 
                                           mesh.vertex_stride_bytes :
                                           3*sizeof(float);
      glBufferData(GL_ARRAY_BUFFER, vertex_count*vertex_stride_bytes, positions, GL_STATIC_DRAW);

      // Fill index buffer

      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh_ebos[meshIdx]);
      const size_t triangle_count = mesh.num_triangles;
      const unsigned int* indices = mesh.tri_vertex_indices; 
      glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int)*triangle_count*3, indices, GL_STATIC_DRAW);
    }

    // Per instance data (vertex array objects, occlusion values)
    m_vaos.resize(m_num_instances);
    glGenVertexArrays((GLsizei)m_num_instances, &m_vaos[0]);
    for (size_t instanceIdx = 0; instanceIdx < m_num_instances; ++instanceIdx) {

      glBindVertexArray(m_vaos[instanceIdx]);
      
      // Bind existing position buffer to shader.
      unsigned mesh_index = m_instances[instanceIdx].mesh_index;
      glBindBuffer(GL_ARRAY_BUFFER, mesh_vbos[mesh_index]);

      const bake::Mesh& mesh = m_meshes[mesh_index];
      const unsigned vertex_stride_bytes = mesh.vertex_stride_bytes > 0 ? 
                                           mesh.vertex_stride_bytes :
                                           3*sizeof(float);
      glVertexAttribPointer(/*slot*/ 0, /*components*/ 3, GL_FLOAT, GL_FALSE, vertex_stride_bytes, /*offset*/ 0);
      glEnableVertexAttribArray(0);

      // Fill occlusion buffer and bind to shader.  This buffer is per instance.
      GLuint occl_vbo = 0;
      glGenBuffers(1, &occl_vbo);
      glBindBuffer(GL_ARRAY_BUFFER, occl_vbo);
      glBufferData(GL_ARRAY_BUFFER, sizeof(float)*mesh.num_vertices, m_vertex_ao[instanceIdx], GL_STATIC_DRAW);
      glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, /*stride*/ 0, /*offset*/ 0);
      glEnableVertexAttribArray(1);
      
      // Bind existing index buffer
      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh_ebos[mesh_index]);

    }

    glEnable(GL_DEPTH_TEST);

    if (m_draw_edges) {
      // Offset filled triangles to reduce Z fighting with edge lines
      glEnable ( GL_POLYGON_OFFSET_FILL );
    }
    glPolygonOffset ( 1, 1 );

    return true;
  }

  virtual void display()
  {
    WindowInertiaCamera::display();

    m_prog.enable();
    mat4f world2screen = m_projection * m_camera.m4_view;
    m_prog.setUniformMatrix4fv("world2screen", world2screen.mat_array, /*transpose*/ false);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);


    for (size_t i = 0; i < m_num_instances; ++i) {

      // Note: optix matrix is transposed from opengl
      m_prog.setUniformMatrix4fv("object2world", const_cast<GLfloat*>(m_instances[i].xform), /*transpose*/ true);
      
      glBindVertexArray(m_vaos[i]);
      glEnableVertexAttribArray(1);  // occlusion attrib

      const bake::Mesh& mesh = m_meshes[m_instances[i].mesh_index];
      const size_t num_triangles = mesh.num_triangles;
      const GLsizei num_indices = static_cast<GLsizei>(num_triangles*3);
      glDrawElements(GL_TRIANGLES, num_indices, GL_UNSIGNED_INT, 0);
    }

    if (m_draw_edges) {

      glPolygonMode ( GL_FRONT_AND_BACK, GL_LINE );	
      for (size_t i = 0; i < m_num_instances; ++i) {

        // Note: optix matrix is transposed from opengl
        m_prog.setUniformMatrix4fv("object2world", const_cast<GLfloat*>(m_instances[i].xform), /*transpose*/ true);
        
        glBindVertexArray(m_vaos[i]);
        // replace occlusion array with constant value for edges
        glDisableVertexAttribArray(1);  
        glVertexAttrib1f(1, 0.2f);

        const bake::Mesh& mesh = m_meshes[m_instances[i].mesh_index];
        const size_t num_triangles = mesh.num_triangles;
        const GLsizei num_indices = static_cast<GLsizei>(num_triangles*3);
        glDrawElements(GL_TRIANGLES, num_indices, GL_UNSIGNED_INT, 0);
      }
      glPolygonMode ( GL_FRONT_AND_BACK, GL_FILL );

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
      case 'e':  // toggle edges
      case 'E':
        m_draw_edges = !m_draw_edges;
        if (m_draw_edges) {
          glEnable ( GL_POLYGON_OFFSET_FILL );
        } else {
          glDisable( GL_POLYGON_OFFSET_FILL );
        }
        break;
    }
  }
  
}; // MyWindow



namespace bake {

  void view( const Mesh* meshes, const size_t num_meshes, 
             const bake::Instance* instances, const size_t num_instances, float const* const* vertex_colors,
             float scene_bbox_min[3], float scene_bbox_max[3])
  {

    vec3f bbox_min(scene_bbox_min);
    vec3f bbox_max(scene_bbox_max);
    const vec3f center = 0.5f*(bbox_min + bbox_max);
    const vec3f bbox_extents = bbox_max - bbox_min;
    const float max_extent = std::max( std::max(bbox_extents[0], bbox_extents[1]), bbox_extents[2]);

    // Initial camera params
    const vec3f eye = center + vec3f(0, 0, 2.0f*max_extent);
    const vec3f lookat = center;
    const float fov = 30.0f;
    const float clipnear = 0.01f*max_extent;
    const float clipfar = 10.0f*max_extent;

    static MyWindow window(meshes, num_meshes, instances, num_instances, vertex_colors, eye, lookat, fov, clipnear, clipfar);

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



