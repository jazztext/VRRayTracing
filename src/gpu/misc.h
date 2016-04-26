#ifndef __MISC_H__
#define __MISC_H__

#include <stdio.h>

namespace VRRT {

#ifdef __CUDACC__ //nvcc

static const double PI = 3.14159;

#define DEBUG

#ifdef DEBUG
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


//#define __STRICT_ANSI__ //solves problems with __float128, unsupported by nvcc

#else //gcc
#define __host__
#define __device__

#endif

__device__
static inline double infy()
{
  #ifdef __CUDACC__
  return __longlong_as_double(0x7ff0000000000000);
  #else
  return INF_D;
  #endif
}


}

#endif
