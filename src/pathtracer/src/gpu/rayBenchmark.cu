#include "ray.h"
#include "bvhGPU.h"
#include "../bvh.h"
#include "../cycleTimer.h"
#include "pathtracer.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cuda_profiler_api.h>

namespace VRRT {

extern __constant__ constantParams cuGlobals;

__global__ void initCurand2(curandState *state)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int ind = y * gridDim.x * blockDim.x + x;
  curand_init(1234, ind, 0, &state[ind]);
}

__device__ Vector3D pointInBox(BBox bbox, curandState *state)
{
  float x = curand_uniform(state),
        y = curand_uniform(state),
        z = curand_uniform(state);
  Vector3D p;
  p = Vector3D::make(x * bbox.extent.v.x, y * bbox.extent.v.y, z * bbox.extent.v.z);
  return p + bbox.min;
}

__global__ void genRays(Ray *rays, int numRays, curandState *states, BBox bbox)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= numRays) return;
  curandState localState = states[i];
  Vector3D p1 = pointInBox(bbox, &localState);
  Vector3D p2 = pointInBox(bbox, &localState);
  Vector3D d = p2 - p1;
  Vector3D o = p1 - bbox.extent.norm() * d;
  rays[i] = Ray(o, d);
}

__global__ void raycast(Ray *rays, int numRays, BVHGPU bvh)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= numRays) return;
  Ray r = rays[i];
  Intersection inters;
  bvh.intersect(r, &inters);
}

void benchmark(CMU462::StaticScene::BVHAccel *bvh, int numRays)
{
  Vector3D *points, *normals;
  BVHGPU bvhGPU(bvh, &points, &normals);
  constantParams params;
  params.points = points;
  params.normals = normals;
  cudaCheckError( cudaMemcpyToSymbol(cuGlobals, &params, sizeof(constantParams)) );

  dim3 blockDim(256);
  dim3 gridDim((numRays + blockDim.x - 1) / blockDim.x);

  //init curand state
  curandState *states;
  int stateSize = blockDim.x * blockDim.y * gridDim.x * gridDim .y;
  cudaCheckError( cudaMalloc(&states, sizeof(curandState) * stateSize) );
  initCurand2<<<gridDim, blockDim>>>(states);
  cudaCheckError( cudaGetLastError() );
  cudaCheckError( cudaDeviceSynchronize() );

  //generate rays
  Ray *rays;
  cudaCheckError( cudaMalloc(&rays, sizeof(Ray) * numRays) );
  CMU462::BBox bbox = bvh->get_bbox();
  BBox bbox2(bbox.min, bbox.max);
  genRays<<<gridDim, blockDim>>>(rays, numRays, states, bbox2);
  cudaCheckError( cudaGetLastError() );
  cudaCheckError( cudaDeviceSynchronize() );

  //run test
  std::cout << "Casting rays... ";
  fflush(stdout);
  cudaProfilerStart();
  double start = CycleTimer::currentSeconds();
  raycast<<<gridDim, blockDim>>>(rays, numRays, bvhGPU);
  cudaCheckError( cudaGetLastError() );
  cudaCheckError( cudaDeviceSynchronize() );
  double end = CycleTimer::currentSeconds();
  cudaProfilerStop();
  std::cout << "Done! (" << end - start << " sec)\n";

}

}
