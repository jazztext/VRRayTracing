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


namespace VRRT {

__device__ inline unsigned char colorToChar(float c)
{
  c *= 255;
  if (c > 255) return 255;
  else if (c < 0) return 0;
  else return (unsigned char) c;
}

__device__ Spectrum trace_ray(Ray r,
                              SceneLight *lights,
                              int numLights,
                              BVHGPU *bvh, int ns_area_light, int max_ray_depth,
                              curandState *state, bool includeLe = false) {
  Spectrum total, multiplier(1, 1, 1);
  while (1) {

  if (r.depth > max_ray_depth) return total;
  Intersection isect;
  //check for intersection
  if (!bvh->intersect(r, &isect)) {
    return total;
    // Environment lighting goes here...
  }

  //initialize L_out with emission from intersected material, if applicable
  Spectrum L_out = includeLe ? isect.bsdf->get_emission() : Spectrum();

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

  for (int n = 0; n < numLights; n++) {
    SceneLight &light = lights[n];
    int num_light_samples = light.is_delta_light() ? 1 : ns_area_light;

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

      Ray shadow = Ray(hit_p + .0001f*dir_to_light, dir_to_light);
      Intersection shadowIsect;
      if (!bvh->intersect(shadow, &shadowIsect) ||
          shadowIsect.t > dist_to_light)
        L_out += f * light_L * cos_theta * (scale / pdf);
    }
  }

  total += multiplier * L_out;

  Vector3D w_i;
  float pdf2;
  bool inMaterial = r.inMaterial;
  Spectrum s = isect.bsdf->sample_f(w_out, &w_i, &pdf2, inMaterial, state);
  w_i = w2o.inv() * w_i;
  w_i.normalize();
  float killP = 1.f - s.illum();
  killP = clamp(killP, 0.0f, 1.0f);
  if (UniformGridSampler2D().get_sample(state).x < killP) return total;
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

__global__ void raytrace_pixel(unsigned char *img, CMU462::Camera c,
                               SceneLight *lights,
                               int numLights, BVHGPU bvh, int h, int w,
                               int ns_aa, int ns_area_light, int max_ray_depth,
                               curandState *state)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;
  int id = y * gridDim.x * blockDim.x + x;
  curandState localState = state[id];

  Spectrum total = Spectrum();
  if (ns_aa > 1) {
    UniformGridSampler2D gridSampler;
    for (int i = 0; i < ns_aa; i++) {
      Vector2D p = gridSampler.get_sample(&localState);
      Ray r = c.generate_ray((x + p.x) / w, (y + p.y) / h);
      total += trace_ray(r, lights, numLights, &bvh, ns_area_light,
                         max_ray_depth, &localState, true);
    }
    total *= (1.0 / ns_aa);
  } else {
    Ray r = c.generate_ray((x + 0.5) / w, (y + 0.5) / h);
    total = trace_ray(r, lights, numLights, &bvh, ns_area_light, max_ray_depth,
                      &localState, true) * (1.f / ns_aa);
  }
  int ind = 4 * (x + y * w);
  img[ind]     = colorToChar(total.r);
  img[ind + 1] = colorToChar(total.g);
  img[ind + 2] = colorToChar(total.b);
  state[id] = localState;
}

__global__ void initCurand(curandState *state)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int ind = y * gridDim.x * blockDim.x + x;
  curand_init(1234, ind, 0, &state[ind]);
}

__global__ void initBuffer(unsigned char *outBuffer, int h, int w)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;
  for (int i = 0; i < 4; i++) {
    outBuffer[i + 4 * (x + w * y)] = 255*(i==3);
  }
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
                    int max_ray_depth)
{
  setup();
  dim3 blockDim(256);
  dim3 gridDim((screenW + blockDim.x - 1) / blockDim.x,
               (screenH + blockDim.y - 1) / blockDim.y);
  unsigned char *outBuffer, *frame_out = new unsigned char[4*screenW*screenH];
  //initialize output buffer on GPU
  cudaCheckError( cudaMalloc(&outBuffer, sizeof(unsigned char) * screenW * screenH * 4) );
  initBuffer<<<gridDim, blockDim>>>(outBuffer, screenH, screenW);
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
  BVHGPU bvhGPU(bvh);
  float end = CycleTimer::currentSeconds();
  std::cout << "Done! (" << end - start << " sec)\n";

  //copy lights over to GPU
  std::cout << "Copying Lights to GPU... ";
  fflush(stdout);
  start = CycleTimer::currentSeconds();
  SceneLight *lights;
  cudaCheckError( cudaMalloc(&lights, sizeof(SceneLight) * scene->lights.size()) );
  cudaCheckError( cudaMemcpy(lights, scene->lights.data(),
                             sizeof(SceneLight) * scene->lights.size(),
                             cudaMemcpyHostToDevice) );

  end = CycleTimer::currentSeconds();
  std::cout << "Done! (" << end - start << " sec)\n";

  //raytrace scene
  std::cout << "Raytracing scene... ";
  fflush(stdout);
  cudaProfilerStart();
  start = CycleTimer::currentSeconds();
  raytrace_pixel<<<gridDim, blockDim>>>(outBuffer, *c, lights,
                                        scene->lights.size(), bvhGPU, screenH,
                                        screenW, ns_aa, ns_area_light,
                                        max_ray_depth, state);
  cudaCheckError( cudaGetLastError() );
  cudaCheckError( cudaDeviceSynchronize() );
  end = CycleTimer::currentSeconds();
  cudaProfilerStop();
  std::cout << "Done! (" << end - start << " sec)\n";

  //copy image into CPU buffer
  cudaCheckError( cudaMemcpy(frame_out, outBuffer, 4*screenW*screenH, cudaMemcpyDeviceToHost) );

  //write image to file
  lodepng::encode("Raytraced.png", frame_out, screenW, screenH);
  printf("success\n");

  //free memory
  delete frame_out;
  cudaCheckError( cudaFree(state) );
  cudaCheckError( cudaFree(outBuffer) );
  //for (int i = 0; i < scene->lights.size(); i++) {
  //  cudaFree(cpuLights[i]);
  //}
  cudaFree(lights);
  cudaDeviceReset();
}

}
