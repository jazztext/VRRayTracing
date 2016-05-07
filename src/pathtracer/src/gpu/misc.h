#ifndef __MISC_H__
#define __MISC_H__

#include <stdio.h>
#include <curand.h>

namespace VRRT {

#ifdef __CUDACC__ //nvcc

static const float PI = 3.14159;

#define DEBUG_

#ifdef DEBUG_
#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(code),
            file, line);
    if (abort) exit(code);
  }
}
#else
#define cudaCheckError(ans) ans
#endif

#else //gcc
//#define __host__
//#define __device__

#endif

__device__
static inline float infy()
{
  #ifdef __CUDACC__
  //return __int_as_float(0x7f800000);
  return 1e15;
  #else
  return INF_F;
  #endif
}


}

#endif
