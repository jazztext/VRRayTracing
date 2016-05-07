#ifndef __PATHTRACER_H__
#define __PATHTRACER_H__

#include <CMU462/CMU462.h>
#include <CMU462/viewer.h>

/*
#define TINYEXR_IMPLEMENTATION
#include <CMU462/tinyexr.h>
*/

#include <gpu/pathtracer.h>
#include <application.h>
#include <gpu/bvhGPU.h>


void usage(const char *binaryName);

namespace VRRT {

__global__ void initCurand(curandState *state);
__global__ void initBuffer(unsigned char *outbuffer, int h, int w);
__global__ void raytrace_pixel(unsigned char*, CMU462::Camera, VRRT::SceneLight*, int, VRRT::BVHGPU bvh, int,int,int,int,int,curandState*);

}
#endif // __PATHTRACER_H__
