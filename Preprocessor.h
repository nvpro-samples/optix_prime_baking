//
//Copyright (c) 2013 NVIDIA Corporation.  All rights reserved.
//
//NVIDIA Corporation and its licensors retain all intellectual property and
//proprietary rights in and to this software, related documentation and any
//modifications thereto.  Any use, reproduction, disclosure or distribution of
//this software and related documentation without an express license agreement
//from NVIDIA Corporation is strictly prohibited.
//
//TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
//*AS IS* AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
//OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF
//MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL
//NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR
//CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR
//LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF BUSINESS
//INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
//INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE
//POSSIBILITY OF SUCH DAMAGES
//

#pragma once

#include <optix_prime/optix_prime.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>


//------------------------------------------------------------------------------
#define CHK_CUDA( code )                                                       \
{                                                                              \
  cudaError_t err__ = code;                                                    \
  if( err__ != cudaSuccess )                                                   \
  {                                                                            \
    std::cerr << "Error on line " << __LINE__ << ":"                           \
              << cudaGetErrorString( err__ ) << std::endl;                     \
    exit(1);                                                                   \
  }                                                                            \
}

//------------------------------------------------------------------------------
#define CHK_PRIME( code )                                                      \
{                                                                              \
  RTPresult res__ = code;                                                      \
  if( res__ != RTP_SUCCESS )                                                   \
  {                                                                            \
  const char* err_string;                                                      \
  rtpContextGetLastErrorString( context, &err_string );                        \
  std::cerr << "Error on line " << __LINE__ << ": '"                           \
  << err_string                                                                \
  << "' (" << res__ << ")" << std::endl;                                       \
  exit(1);                                                                     \
  }                                                                            \
}

