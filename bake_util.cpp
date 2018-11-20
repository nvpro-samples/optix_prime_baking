/* Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "bake_util.h"
#include <iomanip>
#include <iostream>

// System timing code copied from OptiX SDK
#if defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    include<windows.h>
#    include<mmsystem.h>
#else /*Apple and Linux both use this */
#    include<sys/time.h>
#    include <unistd.h>
#    include <dirent.h>
#endif

#if defined(_WIN32)

// inv_freq is 1 over the number of ticks per second.
static double inv_freq;
static int freq_initialized = 0;
static int use_high_res_timer = 0;

bool sutilCurrentTime( double* current_time )
{
  if (!freq_initialized) {
    LARGE_INTEGER freq;
    use_high_res_timer = QueryPerformanceFrequency( &freq );
    inv_freq = 1.0/freq.QuadPart;
    freq_initialized = 1;
  }
  if (use_high_res_timer) {
    LARGE_INTEGER c_time;
    if (QueryPerformanceCounter( &c_time )) {
      *current_time = c_time.QuadPart*inv_freq;
    } else {
      return false;
    }
  } else {
    *current_time = ((double)timeGetTime( )) * 1.0e-3;
  }
  return true;
}

#else

bool sutilCurrentTime( double* current_time )
{
  struct timeval tv;
  if (gettimeofday( &tv, 0 )) {
    std::cerr << "sutilCurrentTime(): gettimeofday failed!\n";
    return false;
  }

  *current_time = tv.tv_sec+ tv.tv_usec * 1.0e-6;
  return true;
}

#endif

double Timer::start( )
{
  sutilCurrentTime( &t0 ); return t0;
}

double Timer::stop( )
{
  sutilCurrentTime( &t1 ); elapsed += t1-t0; return t1;
}

void printTimeElapsed( Timer& t )
{
  if (t.t1 < t.t0) t.stop();
  std::cerr << std::setw( 8 ) 
            << std::fixed 
            << std::setprecision( 2 ) 
            << t.elapsed * 1000.0 << " ms" << std::endl;
}
