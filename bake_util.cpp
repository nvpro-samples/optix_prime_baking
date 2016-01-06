
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
