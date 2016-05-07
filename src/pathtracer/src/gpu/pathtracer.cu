#include "pathtracer.h"
#include "bsdf.h"
#include "ray.h"
#include "bvhGPU.h"
#include "spectrum.h"
#include "sampler.h"
#include "vector3D.h"
#include "matrix3x3.h"
#include "light.h"
#include "../bvh.h"
#include "../camera.h"
#include "../cycleTimer.h"

#include "CMU462/lodepng.h"
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cuda_profiler_api.h>

#define MAX_LIGHTS 10

namespace VRRT {

__constant__ constantParams cuGlobals;
__constant__ SceneLight cuLights[MAX_LIGHTS];

__device__ inline unsigned char colorToChar(float c)
{
  c *= 255;
  if (c > 255) return 255;
  else if (c < 0) return 0;
  else return (unsigned char) c;
}

__device__ Spectrum trace_ray(Ray r, BVHGPU *bvh, curandState *state,
                              bool includeLe = false) {
  Spectrum total = Spectrum::make(0, 0, 0);
  Spectrum multiplier = Spectrum::make(1, 1, 1);
  while (1) {

  if (r.depth > cuGlobals.max_ray_depth) return total;
  Intersection isect;
  //check for intersection
  if (!bvh->intersect(r, &isect)) {
    return total;
    // Environment lighting goes here...
  }

  //initialize L_out with emission from intersected material, if applicable
  Spectrum L_out = includeLe ? isect.bsdf->get_emission() : Spectrum::make();

  Vector3D hit_p = r.o + r.d * isect.t;
  Vector3D hit_n = isect.n;

  // make a coordinate system for a hit point
  // with N aligned with the Z direction.
  Matrix3x3 o2w;
  make_coord_space(o2w, isect.n);
  Matrix3x3 w2o = o2w.T();

  // w_out points towards the source of the ray (e.g.,
  // toward the camera if this is a primary ray)
  Vector3D w_out = w2o * (r.o - hit_p);
  w_out.normalize();

  Vector3D dir_to_light;
  float dist_to_light;
  float pdf;

  for (int n = 0; n < cuGlobals.numLights; n++) {
    SceneLight &light = cuLights[n];
    int num_light_samples = light.is_delta_light() ? 1 : cuGlobals.ns_area_light;

    // integrate light over the hemisphere about the normal
    float scale = 1.f / num_light_samples;
    for (int i=0; i<num_light_samples; i++) {

      // returns a vector 'dir_to_light' that is a direction from
      // point hit_p to the point on the light source.  It also returns
      // the distance from point x to this point on the light source.
      Spectrum light_L = light.sample_L(hit_p, &dir_to_light, &dist_to_light, &pdf, state);

      // convert direction into coordinate space of the surface, where
      // the surface normal is [0 0 1]
      Vector3D w_in = w2o * dir_to_light;

      float cos_theta = fmaxf(0.f, w_in[2]);

      // evaluate surface bsdf
      Spectrum f = isect.bsdf->f(w_out, w_in);

      Ray shadow = Ray(hit_p + .00001f*dir_to_light, dir_to_light);
      Intersection shadowIsect;
      if (!bvh->intersect(shadow, &shadowIsect) ||
          shadowIsect.t > dist_to_light)
        L_out += f * light_L * cos_theta * (scale / pdf);
    }
  }

  total +=  multiplier * L_out;

  Vector3D w_i;
  float pdf2;
  bool inMaterial = r.inMaterial;
  Spectrum s = isect.bsdf->sample_f(w_out, &w_i, &pdf2, inMaterial, state);
  w_i = w2o.inv() * w_i;
  w_i.normalize();
  float killP = 1.f - (multiplier).illum();
  killP = clamp(killP, 0.0f, 1.0f);
  if (curand_uniform(state) < killP) return total;
  Ray newR = Ray(hit_p + EPS_F*w_i, w_i);
  newR.depth = r.depth + 1;
  newR.min_t = 0.0;
  newR.max_t = infy();
  newR.inMaterial = inMaterial;
  multiplier *= s * (fabsf(dot(w_i, hit_n)) / (pdf2 * (1 - killP)));
  r = newR;
  includeLe = isect.bsdf->is_delta();
  }
}

__global__ void raytrace_pixel(unsigned int *img, CMU462::Camera c, BVHGPU bvh,
                               curandState *state)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= cuGlobals.w || y >= cuGlobals.h) return;
  int id = y * gridDim.x * blockDim.x + x;
  curandState localState = state[id];

  Spectrum total = Spectrum();
  if (cuGlobals.ns_aa > 1) {
    for (int i = 0; i < cuGlobals.ns_aa; i++) {
      Vector2D p = uniformGridSample(&localState);
      Ray r = c.generate_ray((x + p.x) / cuGlobals.w, (y + p.y) / cuGlobals.h);
      total += trace_ray(r, &bvh, &localState, true);
    }
    total *= (1.0 / cuGlobals.ns_aa);
  } else {
    Ray r = c.generate_ray((x + 0.5) / cuGlobals.w, (y + 0.5) / cuGlobals.h);
    total = trace_ray(r, &bvh, &localState, true);
  }
  int ind = (x + y * cuGlobals.w);
  //weird union stuff to make sure all 4 bytes are copied over at once
  union {
    unsigned char channels[4];
    unsigned int p;
  };
  channels[0] = colorToChar(total.r);
  channels[1] = colorToChar(total.g);
  channels[2] = colorToChar(total.b);
  channels[3] = 255;
  img[ind] = p;
  //state[id] = localState;
}

__global__ void initCurand(curandState *state)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int ind = y * gridDim.x * blockDim.x + x;
  curand_init(1234, ind, 0, &state[ind]);
}

__global__ void initBuffer(unsigned int *outBuffer)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= cuGlobals.w || y >= cuGlobals.h) return;
  union {
    unsigned char channels[4];
    unsigned int p;
  };
  for (int i = 0; i < 4; i++) {
    channels[i] = 255*(i==3);
  }
  outBuffer[x + cuGlobals.w * y] = p;
}

void setup()
{
  int deviceCount = 0;
  std::string name;
  cudaError_t err = cudaGetDeviceCount(&deviceCount);

  printf("---------------------------------------------------------\n");
  printf("Initializing CUDA\n");
  printf("Found %d CUDA devices\n", deviceCount);

  for (int i=0; i<deviceCount; i++) {
      cudaDeviceProp deviceProps;
      cudaGetDeviceProperties(&deviceProps, i);
      name = deviceProps.name;

      printf("Device %d: %s\n", i, deviceProps.name);
      printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
      printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
      printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
  }
  printf("---------------------------------------------------------\n");

}

void raytrace_scene(CMU462::Camera *c, CMU462::StaticScene::Scene *scene,
                    CMU462::StaticScene::BVHAccel *bvh, int screenW,
                    int screenH, int ns_aa, int ns_area_light,
                    int max_ray_depth, char *fname)
{
  setup();
  dim3 blockDim(256);
  dim3 gridDim((screenW + blockDim.x - 1) / blockDim.x,
               (screenH + blockDim.y - 1) / blockDim.y);
  unsigned int *outBuffer;
  unsigned char *frame_out = new unsigned char[4*screenW*screenH];

  //initialize output buffer on GPU
  cudaCheckError( cudaMalloc(&outBuffer, sizeof(unsigned int) * screenW * screenH) );
  initBuffer<<<gridDim, blockDim>>>(outBuffer);
  cudaCheckError( cudaGetLastError() );
  cudaCheckError( cudaDeviceSynchronize() );

  //initialize curand state for RNG
  curandState *state;
  int stateSize = blockDim.x * blockDim.y * gridDim.x * gridDim .y;
  cudaCheckError( cudaMalloc(&state, sizeof(curandState) * stateSize) );
  initCurand<<<gridDim, blockDim>>>(state);
  cudaCheckError( cudaGetLastError() );
  cudaCheckError( cudaDeviceSynchronize() );

  std::cout << "Constructing BVH and transfer to GPU... ";
  fflush(stdout);
  float start = CycleTimer::currentSeconds();
  //construct bvh on GPU
  Vector3D *points, *normals;
  BVHGPU bvhGPU(bvh, &points, &normals);
  float end = CycleTimer::currentSeconds();
  std::cout << "Done! (" << end - start << " sec)\n";

  //copy parameters to GPU global memory
  constantParams params;
  params.numLights = scene->lights.size();
  params.max_ray_depth = max_ray_depth;
  params.ns_aa = ns_aa;
  params.ns_area_light = ns_area_light;
  params.w = screenW;
  params.h = screenH;
  params.points = points;
  params.normals = normals;
  cudaCheckError( cudaMemcpyToSymbol(cuGlobals, &params, sizeof(constantParams)) );

  //copy lights over to GPU
  std::cout << "Copying Lights to GPU... ";
  fflush(stdout);
  start = CycleTimer::currentSeconds();
  cudaCheckError( cudaMemcpyToSymbol(cuLights, scene->lights.data(),
                                     sizeof(SceneLight) * scene->lights.size()) );

  end = CycleTimer::currentSeconds();
  std::cout << "Done! (" << end - start << " sec)\n";

  //raytrace scene
  std::cout << "Raytracing scene... ";
  fflush(stdout);
  cudaProfilerStart();
  start = CycleTimer::currentSeconds();
  raytrace_pixel<<<gridDim, blockDim>>>(outBuffer, *c, bvhGPU, state);
  cudaCheckError( cudaGetLastError() );
  cudaCheckError( cudaDeviceSynchronize() );
  end = CycleTimer::currentSeconds();
  cudaProfilerStop();
  std::cout << "Done! (" << end - start << " sec)\n";

  //copy image into CPU buffer
  cudaCheckError( cudaMemcpy(frame_out, outBuffer, sizeof(int)*screenW*screenH, cudaMemcpyDeviceToHost) );

  //write image to file
  lodepng::encode(fname, frame_out, screenW, screenH);
  printf("success\n");

  //free memory
  delete frame_out;
  cudaCheckError( cudaFree(state) );
  cudaCheckError( cudaFree(outBuffer) );
  cudaDeviceReset();
}

}
