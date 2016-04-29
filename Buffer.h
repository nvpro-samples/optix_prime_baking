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

#pragma once

#include "Preprocessor.h"
#include <vector>

//------------------------------------------------------------------------------
//
// 
//
enum PageLockedState
{
  UNLOCKED,
  LOCKED
};


//------------------------------------------------------------------------------
//
// A simple abstraction for memory to be passed into Prime via BufferDescs
//
template<typename T>
class Buffer
{
public:
  Buffer( size_t count=0, RTPbuffertype type=RTP_BUFFER_TYPE_HOST, PageLockedState pageLockedState=UNLOCKED, unsigned stride=0 ) 
    : m_ptr( 0 ),
      m_tempHost( 0 ),
      m_pageLockedState( pageLockedState )
  {
    alloc( count, type, pageLockedState, stride );
  }

  // Allocate without changing type or stride
  void alloc( size_t count )
  {
    alloc( count, m_type, m_pageLockedState );
  }

  void alloc( size_t count, RTPbuffertype type, PageLockedState pageLockedState=UNLOCKED, unsigned stride=0 )
  {
    if( m_ptr )
      free();

    m_type = type;
    m_count = count;
    m_stride = stride;
    if( m_count > 0 ) 
    {
      if( m_type == RTP_BUFFER_TYPE_HOST )
      {
        m_ptr = (T*)malloc(sizeInBytes());
        if( pageLockedState )
          rtpHostBufferLock( m_ptr, sizeInBytes() ); // for improved transfer performance
        m_pageLockedState = pageLockedState;
      }
      else
      {
        CHK_CUDA( cudaGetDevice( &m_device ) );
        CHK_CUDA( cudaMalloc( &m_ptr, sizeInBytes() ) );
      }
    }
  }

  void free()
  {
    if( m_ptr && m_type == RTP_BUFFER_TYPE_HOST )
    {
      if( m_pageLockedState ) {
        rtpHostBufferUnlock( m_ptr );
      }
      ::free(m_ptr);
      ::free(m_tempHost);
    }
    else 
    {
      int oldDevice;
      CHK_CUDA( cudaGetDevice( &oldDevice ) );
      CHK_CUDA( cudaSetDevice( m_device ) );
      CHK_CUDA( cudaFree( m_ptr ) );
      CHK_CUDA( cudaSetDevice( oldDevice ) );
    }

    m_ptr = 0;
    m_tempHost = 0;
    m_count = 0;
    m_stride = 0;
  }

  ~Buffer()
  {
    free();
  }

  size_t count()       const { return m_count; }
  size_t sizeInBytes() const { return m_count * (m_stride ? m_stride : sizeof(T)); }
  const T* ptr()       const { return m_ptr; }
  T* ptr()                   { return m_ptr; }
  RTPbuffertype type() const { return m_type; }
  unsigned stride()    const { return m_stride; }

  const T* hostPtr() 
  {
    if( m_type == RTP_BUFFER_TYPE_HOST )
      return m_ptr;

    if (!m_tempHost) m_tempHost = (T*)malloc(sizeInBytes());
    CHK_CUDA( cudaMemcpy( &m_tempHost[0], m_ptr, sizeInBytes(), cudaMemcpyDeviceToHost ) );
    return &m_tempHost[0];
  }

protected:
  RTPbuffertype m_type;
  T* m_ptr;
  int m_device;
  size_t m_count;
  unsigned m_stride;
  PageLockedState m_pageLockedState;
  T* m_tempHost;
  
private:
  Buffer<T>( const Buffer<T>& );            // forbidden
  Buffer<T>& operator=( const Buffer<T>& ); // forbidden
};

