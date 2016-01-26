
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


// This code resamples occlusion computed at face sample points onto vertices using 
// the method from the following paper:
// Least Squares Vertex Baking, L. Kavan, A. W. Bargteil, P.-P. Sloan, EGSR 2011
// 
// Adapted from code written originally by Peter-Pike Sloan.


#include "bake_api.h"
#include "bake_filter_least_squares.h"
#include "bake_util.h"

#include <cassert>
#include <iostream>
#include <vector>
#include <optixu/optixu_math_namespace.h>


#ifdef EIGEN3_ENABLED

#include <Eigen/Core>
#include <Eigen/Geometry>  // for cross product
#include <Eigen/SparseCholesky>

using namespace optix;


typedef double ScalarType;
typedef Eigen::SparseMatrix<ScalarType> SparseMatrix;
typedef Eigen::Matrix<ScalarType, 2, 3> Matrix23;
typedef Eigen::Matrix<ScalarType, 2, 4> Matrix24;
typedef Eigen::Matrix<ScalarType, 4, 4> Matrix44;
typedef Eigen::Triplet<ScalarType> Triplet;
typedef Eigen::Matrix<ScalarType, 3, 1> Vector3;
typedef Eigen::Matrix<ScalarType, 2, 1> Vector2;

namespace {

ScalarType triangleArea(const Vector3 &a, const Vector3 &b, const Vector3 &c)
{
	Vector3 ba = b - a, ca = c - a;
	Vector3 crop = ba.cross(ca);
	return crop.norm() * 0.5f;
}

// Embeds 3D triangle v[0], v[1], v[2] into a plane, such that:
//  p[0] = (0, 0), p[1] = (0, positive number), p[2] = (positive number, any number)
// If triangle is close to degenerate returns false and p is undefined.
bool planarizeTriangle(const Vector3 v[3], Vector2 p[3])
{
	double l01 = (v[0] - v[1]).norm();
	double l02 = (v[0] - v[2]).norm();
	double l12 = (v[1] - v[2]).norm();

  const double eps = 0.0;
  if (l01 <= eps || l02 <= eps || l12 <= eps) return false;

	double p2y = (l02*l02 + l01*l01 - l12*l12) / (2.0 * l01);
	double tmp1 = l02*l02 - p2y*p2y;
  if (tmp1 <= eps) return false;

	p[0] = Vector2(0.0f, 0.0f);
	p[1] = Vector2(0.0f, l01);
	p[2] = Vector2(sqrt(tmp1), p2y);
	return true;
}

// Computes gradient operator (2 x 3 matrix 'grad') for a planar triangle.  If
// 'normalize' is false then division by determinant is off (and thus the
// routine cannot fail even for degenerate triangles).
bool triGrad2D(const Vector2 p[3], const bool normalize, Matrix23 &grad)
{
	double det = 1.0;
	if (normalize) {
    det = -double(p[0](1))*p[1](0) + double(p[0](0))*p[1](1) + double(p[0](1))*p[2](0) 
          - double(p[1](1))*p[2](0) - double(p[0](0))*p[2](1) + double(p[1](0))*p[2](1);	
    const double eps = 0.0;
    if (fabs(det) <= eps) {
	  	return false;
	  }
  }

	grad(0,0) = p[1](1) - p[2](1);
	grad(0,1) = p[2](1) - p[0](1);
	grad(0,2) = p[0](1) - p[1](1);

	grad(1,0) = p[2](0) - p[1](0);
	grad(1,1) = p[0](0) - p[2](0);
	grad(1,2) = p[1](0) - p[0](0);

	grad /= det;
	return true;
}

// Computes difference of gradients operator (2 x 4 matrix 'GD') for a butterfly, i.e., 
// two edge-adjacent triangles.
// Performs normalization so that units are [m], so GD^T * GD will have units of area [m^2]:
bool butterflyGradDiff(const Vector3 v[4], Matrix24 &GD)
{
	Vector3 v1[3] = {v[0], v[1], v[2]};
	Vector3 v2[3] = {v[0], v[1], v[3]};
	Vector2 p1[3], p2[3];	
	bool success = planarizeTriangle(v1, p1);
	if (!success) return false;
	success = planarizeTriangle(v2, p2);
	if (!success) return false;
	p2[2](0) *= -1.0; // flip the x coordinate of the last vertex of the second triangle so we get a butterfly

	Matrix23 grad1, grad2;	
	success = triGrad2D(p1, /*normalize*/ true, grad1);
	if (!success) return false;
	success = triGrad2D(p2, true, grad2);
	if (!success) return false;

	Matrix24 gradExt1, gradExt2;
	for (int i=0; i<2; i++) {
		gradExt1(i, 0) = grad1(i, 0);  gradExt1(i, 1) = grad1(i, 1);  gradExt1(i, 2) = grad1(i, 2); gradExt1(i, 3) = 0.0;
		gradExt2(i, 0) = grad2(i, 0);  gradExt2(i, 1) = grad2(i, 1);  gradExt2(i, 2) = 0.0;         gradExt2(i, 3) = grad2(i, 2);
	}
	GD = gradExt1 - gradExt2;

	const ScalarType area1 = triangleArea(v1[0], v1[1], v1[2]);
	const ScalarType area2 = triangleArea(v2[0], v2[1], v2[2]);
	GD *= (area1 + area2);
		
	return true;
}

struct Butterfly {
  std::pair<int, int> wingverts;
  int count;
  Butterfly() 
    : wingverts(std::make_pair(-1, -1)), count(0)
  {}
};

typedef std::map<std::pair<int, int>, Butterfly > EdgeMap;

const float3* get_vertex(const float* v, unsigned stride_bytes, int index)
{
  return reinterpret_cast<const float3*>(reinterpret_cast<const unsigned char*>(v) + index*stride_bytes);
}

void edgeBasedRegularizer(const float* verts, size_t num_verts, unsigned vertex_stride_bytes, const int3* faces, const size_t num_faces,
  SparseMatrix &regularization_matrix)
{
  const unsigned stride_bytes = vertex_stride_bytes > 0 ? vertex_stride_bytes : 3*sizeof(float);

  // Build edge map.  Each non-boundary edge stores the two opposite "butterfly" vertices that do not lie on the edge.
  EdgeMap edges;

  for (size_t i = 0; i < num_faces; ++i) {
    const int indices[] = {faces[i].x, faces[i].y, faces[i].z};
    for (int k = 0; k < 3; ++k) {
      const int index0 = std::min(indices[k], indices[(k+1)%3]);
      const int index1 = std::max(indices[k], indices[(k+1)%3]);
      const std::pair<int, int> edge = std::make_pair(index0, index1);
      
      if (index0 == indices[k]) {
        edges[edge].wingverts.first = indices[(k+2)%3];  // butterfly vert on left side of edge, ccw
        edges[edge].count++;
      } else {
        edges[edge].wingverts.second = indices[(k+2)%3];  // and right side 
        edges[edge].count++;
      }

    }
  }
  
  size_t skipped = 0;

  std::vector< Triplet > triplets;
  size_t edge_index = 0;
  for (EdgeMap::const_iterator it = edges.begin(); it != edges.end(); ++it, ++edge_index) {
    if (it->second.count != 2) {
      continue;  // not an interior edge, ignore
    }

    int vertIdx[4] = {it->first.first, it->first.second, it->second.wingverts.first, it->second.wingverts.second};
    if (it->second.wingverts.first < 0 || it->second.wingverts.second < 0) {
      continue;  // duplicate face, ignore
    }

    Vector3 butterfly_verts[4];
    for (size_t i = 0; i < 4; ++i) {
      const float3* v = get_vertex(verts, stride_bytes, vertIdx[i]);
      butterfly_verts[i] = Vector3(v->x, v->y, v->z);
    }

    Matrix24 GD;
    if (!butterflyGradDiff(butterfly_verts, GD)) {
      skipped++;
      continue;
    }

    Matrix44 GDtGD = GD.transpose() * GD; // units will now be [m^2]

    // scatter GDtGD:
    for (int i=0; i < 4; i++) {
      for (int j=0; j < 4; j++) {
        triplets.push_back(Triplet(vertIdx[i], vertIdx[j], GDtGD(i, j)));
      }
    }
  }

	regularization_matrix.resize((int)num_verts, (int)num_verts);
	regularization_matrix.setFromTriplets(triplets.begin(), triplets.end());

  if (skipped > 0) {
    std::cerr << "edgeBasedRegularizer: skipped " << skipped << " edges out of " << edges.size() << std::endl;
  }

}


void build_regularization_matrix(
    const bake::Mesh& mesh,
    SparseMatrix& regularization_matrix,
    Timer& timer
  )
{

  timer.start();
  const int3* tri_vertex_indices  = reinterpret_cast<int3*>( mesh.tri_vertex_indices );
  edgeBasedRegularizer(mesh.vertices, mesh.num_vertices, mesh.vertex_stride_bytes, tri_vertex_indices, mesh.num_triangles, regularization_matrix);
  timer.stop();
}


void filter_mesh_least_squares(
    const bake::Mesh&       mesh,
    const bake::AOSamples&  ao_samples,
    const float*            ao_values,
    const float             regularization_weight,
    const SparseMatrix&     regularization_matrix,
    float*                  vertex_ao,
    Timer&                  mass_matrix_timer,
    Timer&                  decompose_timer,
    Timer&                  solve_timer
    )
{
  std::fill(vertex_ao, vertex_ao + mesh.num_vertices, 0.0f);

  mass_matrix_timer.start();

  std::vector< Triplet > triplets;
  triplets.reserve(ao_samples.num_samples * 9);
  const int3* tri_vertex_indices  = reinterpret_cast<int3*>( mesh.tri_vertex_indices );

  for (size_t i = 0; i < ao_samples.num_samples; ++i) {
    const bake::SampleInfo& info = ao_samples.sample_infos[i];
    const int3& tri = tri_vertex_indices[info.tri_idx];

    const float val = ao_values[i] * info.dA;

    vertex_ao[tri.x] += info.bary[0] * val;
    vertex_ao[tri.y] += info.bary[1] * val;
    vertex_ao[tri.z] += info.bary[2] * val;

    // Note: the reference paper suggests computing the mass matrix analytically.
    // Building it from samples gave smoother results for low numbers of samples per face.
    
    triplets.push_back( Triplet( tri.x, tri.x, static_cast<ScalarType>( info.bary[0]*info.bary[0]*info.dA ) ) );
    triplets.push_back( Triplet( tri.y, tri.y, static_cast<ScalarType>( info.bary[1]*info.bary[1]*info.dA ) ) );
    triplets.push_back( Triplet( tri.z, tri.z, static_cast<ScalarType>( info.bary[2]*info.bary[2]*info.dA ) ) );

    {
      const double elem = static_cast<ScalarType>(info.bary[0]*info.bary[1]*info.dA);
      triplets.push_back( Triplet( tri.x, tri.y, elem ) );
      triplets.push_back( Triplet( tri.y, tri.x, elem ) );
    }

    {
      const double elem = static_cast<ScalarType>(info.bary[1]*info.bary[2]*info.dA);
      triplets.push_back( Triplet( tri.y, tri.z, elem ) );
      triplets.push_back( Triplet( tri.z, tri.y, elem ) );
    }

    {
      const double elem = static_cast<ScalarType>(info.bary[2]*info.bary[0]*info.dA);
      triplets.push_back( Triplet( tri.x, tri.z, elem ) );
      triplets.push_back( Triplet( tri.z, tri.x, elem ) );
    }

  }

  // Mass matrix
  SparseMatrix mass_matrix( (int)mesh.num_vertices, (int)mesh.num_vertices );
  mass_matrix.setFromTriplets( triplets.begin(), triplets.end() );

  // Fix missing data due to unreferenced verts
  {
    Eigen::VectorXd ones = Eigen::VectorXd::Constant(mesh.num_vertices, 1.0);
    Eigen::VectorXd lumped = mass_matrix * ones;
    for (int i = 0; i < mesh.num_vertices; ++i) {
      if (lumped(i) <= 0.0) {  // all valid entries in mass matrix are > 0
        mass_matrix.coeffRef(i, i) = 1.0;
      }
    }
  }

  mass_matrix_timer.stop();
  
  Eigen::SimplicialLDLT<SparseMatrix> solver;	

  // Optional edge-based regularization for smoother result, see paper for details
  if (regularization_weight > 0.0f) {

    decompose_timer.start();
    SparseMatrix A = mass_matrix + regularization_weight*regularization_matrix;
    solver.compute(A);
    decompose_timer.stop();

  } else {
    decompose_timer.start();
    solver.compute(mass_matrix);
    decompose_timer.stop();
  }

  solve_timer.start();

  assert( solver.info() == Eigen::Success );

  Eigen::VectorXd b( mesh.num_vertices );
  Eigen::VectorXd x( mesh.num_vertices );
  for (size_t k = 0; k < mesh.num_vertices; ++k) {
    b(k) = vertex_ao[k];
    x(k) = 0.0;
  }

  x = solver.solve(b);

  solve_timer.stop();

  assert( solver.info() == Eigen::Success ); // for debug build
  if ( solver.info() == Eigen::Success ) {
    for (size_t k = 0; k < mesh.num_vertices; ++k) {
      vertex_ao[k] = static_cast<float>(x(k));  // Note: allow out-of-range values
    }
  }
}


} //namespace


void bake::filter_least_squares(
    const Scene&        scene,
    const size_t*       num_samples_per_instance,
    const AOSamples&    ao_samples,
    const float*        ao_values,
    const float         regularization_weight,
    float**             vertex_ao
    )
{

  Timer mass_matrix_timer;
  Timer regularization_matrix_timer;
  Timer decompose_timer;
  Timer solve_timer;

  for (size_t meshIdx = 0; meshIdx < scene.num_meshes; meshIdx++) {

    // Build reg. matrix once, it does not depend on rigid xform per instance
    SparseMatrix regularization_matrix;
    if (regularization_weight > 0.0f) {
      build_regularization_matrix(scene.meshes[meshIdx], regularization_matrix, regularization_matrix_timer);
    }

    // Filter all the instances that point to this mesh
    size_t sample_offset = 0;
    for (size_t i = 0; i < scene.num_instances; ++i) {
      if (scene.instances[i].mesh_index == meshIdx) {
        // Point to samples for this instance
        AOSamples instance_ao_samples;
        instance_ao_samples.num_samples = num_samples_per_instance[i];
        instance_ao_samples.sample_positions = ao_samples.sample_positions + 3*sample_offset;
        instance_ao_samples.sample_normals = ao_samples.sample_normals + 3*sample_offset;
        instance_ao_samples.sample_face_normals = ao_samples.sample_face_normals + 3*sample_offset;
        instance_ao_samples.sample_infos = ao_samples.sample_infos + sample_offset;

        const float* instance_ao_values = ao_values + sample_offset;

        filter_mesh_least_squares(scene.meshes[meshIdx], instance_ao_samples, instance_ao_values, regularization_weight, regularization_matrix,
          vertex_ao[i], mass_matrix_timer, decompose_timer, solve_timer);
      }

      sample_offset += num_samples_per_instance[i];
    }
  }

  std::cerr << "\n\tbuild mass matrices ...           ";  printTimeElapsed( mass_matrix_timer );
  if (regularization_weight > 0.0f) {
    std::cerr << "\tbuild regularization matrices ... ";  printTimeElapsed( regularization_matrix_timer );
  }
  std::cerr << "\tdecompose matrices ...            ";  printTimeElapsed( decompose_timer );
  std::cerr << "\tsolve linear systems ...         ";  printTimeElapsed( solve_timer );
}

#else

#include <stdexcept>

void bake::filter_least_squares(
  const Scene&,
  const size_t*,
  const AOSamples&,
  const float*,
  const float,
  float**
  )
{
  throw std::runtime_error( "filter_least_squares called without Eigen3 support");
}

#endif


