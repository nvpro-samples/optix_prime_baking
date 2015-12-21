
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
#include "bake_ao_optix_prime.h"
#include "bake_filter.h"
#include "bake_filter_least_squares.h"
#include "bake_sample.h"
#include <optixu/optixu_math_namespace.h>
#include <cassert>


using namespace optix;


void bake::computeAO( 
    const Mesh&       mesh,
    const AOSamples&  ao_samples,
    int               rays_per_sample,
    float*            ao_values 
    )
{

  bake::ao_optix_prime( mesh, ao_samples, rays_per_sample, ao_values );

}


void bake::sampleSurface(
    const Mesh& mesh,
    const size_t min_samples_per_triangle,
    AOSamples&  ao_samples
    )
{

  bake::sample_surface_random( mesh, min_samples_per_triangle, ao_samples );

}


void bake::mapAOToVertices(
    const Mesh&            mesh,
    const AOSamples&       ao_samples,
    const float*           ao_values,
    bake::VertexFilterMode mode,
    const float            regularization_weight,
    float*                 vertex_ao
    )
{
    if (mode == VERTEX_FILTER_AREA_BASED) {
      bake::filter( mesh, ao_samples, ao_values, vertex_ao ); 
    } else if (mode == VERTEX_FILTER_LEAST_SQUARES) {
      bake::filter_least_squares( mesh, ao_samples, ao_values, regularization_weight, vertex_ao ); 
    } else {
      assert(0 && "invalid vertex filter mode");
    }
}


void bake::mapAOToTextures(
    )
{
}


