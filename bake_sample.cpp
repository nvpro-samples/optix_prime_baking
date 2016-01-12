
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
#include "bake_sample.h"
#include "bake_sample_internal.h"  // templates
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include "random.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

using namespace optix;

// Ref: https://en.wikipedia.org/wiki/Halton_sequence
template <unsigned int BASE>
float halton(const unsigned int index)
{
  float result = 0.0f;
  const float invBase = 1.0f / BASE;
  float f = invBase;
  unsigned int i = index;
  while( i > 0 ) {
    result += f*( i % BASE );
    i = i / BASE;
    f *= invBase;
  }
  return result;
}

float3 faceforward( const float3& normal, const float3& geom_normal )
{
  if ( optix::dot( normal, geom_normal ) > 0.0f ) return normal;
  return -normal;
}

float3 operator*(const optix::Matrix4x4& mat, const float3& v)
{
  return make_float3(mat*make_float4(v, 1.0f)); 
}


void sample_triangle(const optix::Matrix4x4& xform, const optix::Matrix4x4& xform_invtrans,
                     const int3* tri_vertex_indices, const float3* verts,
                     const int3* tri_normal_indices, const float3* normals,
                     const size_t tri_idx, const size_t tri_sample_count, const double tri_area,
                     float3* sample_positions, float3* sample_norms, float3* sample_face_norms, bake::SampleInfo* sample_infos)
{
  const int3&   tri = tri_vertex_indices[tri_idx];
  const float3& v0 = verts[tri.x];
  const float3& v1 = verts[tri.y];
  const float3& v2 = verts[tri.z];

  const float3 face_normal = optix::normalize( optix::cross( v1-v0, v2-v0 ) );
  float3 n0, n1, n2;
  if (normals && tri_normal_indices) {
    const int3& nindex = tri_normal_indices[tri_idx];
    n0 = faceforward( normals[nindex.x], face_normal );
    n1 = faceforward( normals[nindex.y], face_normal );
    n2 = faceforward( normals[nindex.z], face_normal );
  } else {
    // missing vertex normals, so use face normal.
    n0 = face_normal;
    n1 = face_normal; 
    n2 = face_normal;
  }

  // Random offset per triangle, to shift Halton points
  unsigned seed = tea<4>( (unsigned)tri_idx, (unsigned)tri_idx );
  const float2 offset = make_float2( rnd(seed), rnd(seed) );

  for ( size_t index = 0; index < tri_sample_count; ++index )
  {
    sample_infos[index].tri_idx = (unsigned)tri_idx;
    sample_infos[index].dA = static_cast<float>(tri_area / tri_sample_count);

    // Random point in unit square
    float r1 = offset.x + halton<2>((unsigned)index+1);
    r1 = r1 - (int)r1;
    float r2 = offset.y + halton<3>((unsigned)index+1);
    r2 = r2 - (int)r2;
    assert(r1 >= 0 && r1 <= 1);
    assert(r2 >= 0 && r2 <= 1);

    // Map to triangle. Ref: PBRT 2nd edition, section 13.6.4
    float3& bary = *reinterpret_cast<float3*>(sample_infos[index].bary);
    const float sqrt_r1 = sqrt(r1);
    bary.x = 1.0f - sqrt_r1;
    bary.y = r2*sqrt_r1;
    bary.z = 1.0f - bary.x - bary.y;

    sample_positions[index] = xform*(bary.x*v0 + bary.y*v1 + bary.z*v2);
    sample_norms[index] = optix::normalize(xform_invtrans*( bary.x*n0 + bary.y*n1 + bary.z*n2 ));
    sample_face_norms[index] = optix::normalize(xform_invtrans*face_normal);

  }
}


double triangle_area(const float3& v0, const float3& v1, const float3& v2)
{
  float3 e0 = v1 - v0;
  float3 e1 = v2 - v0;
  float3 c = optix::cross(e0, e1);
  double x = c.x, y = c.y, z = c.z;
  return 0.5*sqrt(x*x + y*y + z*z);
}


class TriangleSamplerCallback
{
public:
  TriangleSamplerCallback(const unsigned int minSamplesPerTriangle,
                  const double* areaPerTriangle)
  : m_minSamplesPerTriangle(minSamplesPerTriangle), 
    m_areaPerTriangle(areaPerTriangle)
    {}
          
  unsigned int minSamples(size_t i) const {
    return m_minSamplesPerTriangle;  // same for every triangle
  }
  double area(size_t i) const {
    return m_areaPerTriangle[i];
  }

private:
  const unsigned int m_minSamplesPerTriangle;
  const double* m_areaPerTriangle;
};


void bake::sample_instance(
    const bake::Instance& instance,
    const size_t min_samples_per_triangle,
    bake::AOSamples&  ao_samples
    )
{

  // Setup access to mesh data
  const bake::Mesh& mesh = *instance.mesh;
  const optix::Matrix4x4 xform(instance.xform);
  const optix::Matrix4x4 xform_invtrans = xform.inverse().transpose();
  assert( ao_samples.num_samples >= mesh.num_triangles*min_samples_per_triangle );
  assert( mesh.vertices               );
  assert( mesh.num_vertices           );
  assert( ao_samples.sample_positions );
  assert( ao_samples.sample_normals   );
  assert( ao_samples.sample_infos     );

  const int3*   tri_vertex_indices  = reinterpret_cast<int3*>( mesh.tri_vertex_indices );
  const float3* verts = reinterpret_cast<float3*>( mesh.vertices );
  const int3*   tri_normal_indices = reinterpret_cast<int3*>( mesh.tri_normal_indices );
  const float3* normals        = reinterpret_cast<float3*>( mesh.normals );

  float3* sample_positions  = reinterpret_cast<float3*>( ao_samples.sample_positions );   
  float3* sample_norms      = reinterpret_cast<float3*>( ao_samples.sample_normals   );   
  float3* sample_face_norms = reinterpret_cast<float3*>( ao_samples.sample_face_normals );
  bake::SampleInfo* sample_infos = ao_samples.sample_infos;


  // Compute triangle areas
  std::vector<double> tri_areas(mesh.num_triangles, 0.0);
  for ( size_t tri_idx = 0; tri_idx < mesh.num_triangles; tri_idx++ ) {
    const int3& tri = tri_vertex_indices[tri_idx];
    const double area = triangle_area(xform*verts[tri.x], xform*verts[tri.y], xform*verts[tri.z]);
    tri_areas[tri_idx] = area;
  }

  // Get sample counts
  std::vector<unsigned int> tri_sample_counts(mesh.num_triangles, 0);
  TriangleSamplerCallback cb(min_samples_per_triangle, &tri_areas[0]);
  distribute_samples_generic(cb, ao_samples.num_samples, mesh.num_triangles, &tri_sample_counts[0]);

  // Place samples
  size_t sample_idx = 0;
  for (size_t tri_idx = 0; tri_idx < mesh.num_triangles; tri_idx++) {
    sample_triangle(xform, xform_invtrans,
      tri_vertex_indices, verts, tri_normal_indices, normals, 
      tri_idx, tri_sample_counts[tri_idx], tri_areas[tri_idx],
      sample_positions+sample_idx, sample_norms+sample_idx, sample_face_norms+sample_idx, sample_infos+sample_idx);
    sample_idx += tri_sample_counts[tri_idx];
  }

  assert( sample_idx == ao_samples.num_samples );

#ifdef DEBUG_MESH_SAMPLES
  for (size_t i = 0; i < ao_samples.num_samples; ++i ) {
    const SampleInfo& info = sample_infos[i];
    std::cerr << "sample info (" << i << "): " << info.tri_idx << ", (" << info.bary[0] << ", " << info.bary[1] << ", " << info.bary[2] << "), " << info.dA << std::endl;
  }
#endif

}

class InstanceSamplerCallback
{
public:
  InstanceSamplerCallback(const unsigned int* minSamplesPerInstance,
                  const double* areaPerInstance)
  : m_minSamplesPerInstance(minSamplesPerInstance), 
    m_areaPerInstance(areaPerInstance)
    {}
          
  unsigned int minSamples(size_t i) const {
    return m_minSamplesPerInstance[i];
  }
  double area(size_t i) const {
    return m_areaPerInstance[i];
  }

private:
  const unsigned int* m_minSamplesPerInstance;
  const double* m_areaPerInstance;
};


size_t bake::distribute_samples(
    const bake::Instance* instances,
    const size_t num_instances,
    const size_t min_samples_per_triangle,
    const size_t requested_num_samples,
    unsigned int* num_samples_per_instance
    )
{

  // Compute min samples per instance
  std::vector<unsigned int> min_samples_per_instance(num_instances);
  size_t num_triangles = 0;
  for (size_t i = 0; i < num_instances; ++i) {
    min_samples_per_instance[i] = min_samples_per_triangle * instances[i].mesh->num_triangles; 
    num_triangles += instances[i].mesh->num_triangles;
  }
  const size_t min_num_samples = min_samples_per_triangle*num_triangles;
  size_t num_samples = std::max(min_num_samples, requested_num_samples);

  // Compute surface area per instance
  std::vector<double> area_per_instance(num_instances, 0.0);
  if (num_samples > min_num_samples) {

    for (size_t idx = 0; idx < num_instances; ++idx) {
      const bake::Mesh& mesh = *instances[idx].mesh;
      const optix::Matrix4x4 xform(instances[idx].xform);
      const int3* tri_vertex_indices  = reinterpret_cast<int3*>( mesh.tri_vertex_indices );
      const float3* verts = reinterpret_cast<float3*>( mesh.vertices );
      for (size_t tri_idx = 0; tri_idx < mesh.num_triangles; ++tri_idx) {
        const int3& tri = tri_vertex_indices[tri_idx];
        double area = triangle_area(xform*verts[tri.x], xform*verts[tri.y], xform*verts[tri.z]);
        area_per_instance[idx] += area;
      }
    }

  }

  // Distribute samples
  InstanceSamplerCallback cb(&min_samples_per_instance[0], &area_per_instance[0]);
  distribute_samples_generic(cb, num_samples, num_instances, num_samples_per_instance);

  return num_samples;

}


