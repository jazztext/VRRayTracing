// Scene.cpp
#include "Scene.h"

#ifdef __APPLE__
#include "opengl/gl.h"
#endif

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

#define _USE_MATH_DEFINES
#include <math.h>

#include <stdlib.h>
#include <string.h>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/intersect.hpp>

#include <GL/glew.h>

#include <GL/freeglut.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cuda_gl_interop.h>
#include <cuda_profiler_api.h>

//#include <helper_functions.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>

#include <vector_types.h>

#include <CMU462/lodepng.h>

#include "Logger.h"

#ifndef SCALE
#define SCALE 1
#endif

extern __constant__ VRRT::constantParams cuGlobals;
extern __constant__ VRRT::SceneLight *cuLights;

inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

Scene::Scene()  {
  image_w = 960 / SCALE, image_h = 1080/SCALE;
  eye_w = 960, eye_h = 1080;
  outFbo = 0;
}

Scene::~Scene()
{
}

void Scene::initCuda() {
  printf("Initializing BVH...\n");
  std::vector<CMU462::StaticScene::Primitive*> primitives;
  for (CMU462::StaticScene::SceneObject *obj : app->scene->objects) {
    const std::vector<CMU462::StaticScene::Primitive*> &obj_prims = obj->get_primitives();
    primitives.reserve(primitives.size() + obj_prims.size());
    primitives.insert(primitives.end(), obj_prims.begin(), obj_prims.end());
  }
  CMU462::StaticScene::BVHAccel *cpuBVH = new CMU462::StaticScene::BVHAccel(primitives, 4);
  VRRT::Vector3D *points, *normals;
  bvh = VRRT::BVHGPU(cpuBVH, &points, &normals);
  delete cpuBVH;

  //copy parameters to GPU global memory
  VRRT::constantParams params;
  params.numLights = app->scene->lights.size();
  params.max_ray_depth = app->max_ray_depth;
  params.ns_aa = app->ns_aa;
  params.ns_area_light = app->ns_area_light;
  params.w = image_w;
  params.h = image_h;
  params.points = points;
  params.normals = normals;
  cudaCheckError( cudaMemcpyToSymbol(cuGlobals, &params, sizeof(VRRT::constantParams)) );


  printf("Done!\n");

  blockDim = dim3(256);
  gridDim = dim3((image_w + blockDim.x - 1) / blockDim.x,
                 (image_h + blockDim.y - 1) / blockDim.y);

  /* Initialize cuRAND state */

  printf("Initializng cuRAND...\n");
  stateSize = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
  cudaCheckError(cudaMalloc(&state, sizeof(curandState) * stateSize));
  printf("Allocated memory, starting kernel now... (state = %p, size = %d)\n", state, stateSize * sizeof(curandState));
  VRRT::initCurand<<<gridDim, blockDim>>>(state);
  printf("Kernel Launched.\n");
  cudaCheckError(cudaGetLastError());
  cudaCheckError(cudaDeviceSynchronize());
  printf("Done!\n");

  printf("Transferring lights over...\n");
  cudaCheckError( cudaMemcpyToSymbol(cuLights, app->scene->lights.data(),
                                     sizeof(VRRT::SceneLight) * app->scene->lights.size()) );
  printf("Done!\n");
}

void deleteVBO(GLuint *fbo, struct cudaGraphicsResource *fbo_res) {
  checkCudaErrors(cudaGraphicsUnregisterResource(fbo_res));

  glBindBuffer(1, *fbo);
  glDeleteBuffers(1, fbo);

  *fbo = 0;
}

void Scene::initGL() {
  eye = 0;
  cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId());

  glGenFramebuffers(2, fbo);
  glGenTextures(2, tex);
  for (int i = 0; i < 2; i++) {
    glBindTexture(GL_TEXTURE_2D, tex[i]);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, image_w, image_h, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo[i]);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, tex[i], 0);
    checkCudaErrors(cudaGraphicsGLRegisterImage(&cuda_fbo_resource[i], tex[i], GL_TEXTURE_2D, cudaGraphicsMapFlagsNone));
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_fbo_resource[i], 0));
    printf("cudaResource was mapped, getting pointer now...\n");
    devRenderbuffer[i] = cudaArray_t();
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&devRenderbuffer[i], cuda_fbo_resource[i], 0, 0));
  }
  checkCudaErrors(cudaMalloc(&devOutput, 4*image_w*image_h));

}

/// Draw the scene(matrices have already been set up).
void Scene::DrawScene(
    const glm::mat4& modelview,
    const glm::mat4& projection,
    const glm::mat4& object) const
{


}


void Scene::RenderForOneEye(const float* pMview, const float* pPersp) const {
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo[eye]);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, outFbo);
    if (!eye) {
      glBlitFramebuffer(0, 0, image_w, image_h, 0, 0, eye_w, eye_h, GL_COLOR_BUFFER_BIT, GL_LINEAR);
    } else {
      glBlitFramebuffer(0, 0, image_w, image_h, eye_w+1, 0, 2*eye_w, eye_h, GL_COLOR_BUFFER_BIT, GL_LINEAR);
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void Scene::runStep(char *fname) {
  image_h = 1080 / SCALE;
  image_w = 960 / SCALE;
  cudaCheckError(cudaMalloc(&devOutput, 4*image_h*image_w));
  VRRT::initBuffer<<<gridDim, blockDim>>>(devOutput);
  cudaCheckError(cudaGetLastError());
  cudaCheckError(cudaDeviceSynchronize());

  VRRT::raytrace_pixel<<<gridDim, blockDim>>>(devOutput, app->camera, bvh,
                                              state);
  cudaCheckError(cudaGetLastError());
  cudaCheckError(cudaDeviceSynchronize());

  unsigned char *frame_out = new unsigned char[4*image_w*image_h];
  cudaCheckError(cudaMemcpy(frame_out, devOutput, 4*image_w*image_h, cudaMemcpyDeviceToHost));
  for (int i = 0; i < image_h  / 2; i++) {
    for (int j = 0; j < image_w*4; j++) {
      unsigned char tmp = frame_out[4*image_w*i + j];
      frame_out[4*image_w*i + j] = frame_out[4*image_w*(image_h - 1 - i) + j];
      frame_out[4*image_w*(image_h - 1 - i) + j] = tmp;
    }
  }

  lodepng::encode(fname, frame_out, image_w, image_h);

  delete frame_out;
  cudaCheckError(cudaFree(devOutput));
}

// Here's where we run CUDA!
void Scene::timestep(double /*absTime*/, double dt) {

//  printf("Entered timestep...\n");

  cudaCheckError(cudaDeviceSynchronize());

  // Actual computation
//  VRRT::initBuffer<<<gridDim, blockDim>>>(devOutput, image_h, image_w);
//  cudaCheckError(cudaGetLastError());
//  cudaCheckError(cudaDeviceSynchronize());

  // Actually run the raytracer...
  VRRT::raytrace_pixel<<<gridDim, blockDim>>>(devOutput, app->camera, bvh,
                                              state);

  cudaCheckError(cudaGetLastError());
  cudaCheckError(cudaDeviceSynchronize());

  // Copy into the FBO...
  cudaCheckError(cudaMemcpyToArray(devRenderbuffer[eye], 0, 0, devOutput, 4*image_w*image_h, cudaMemcpyDeviceToDevice));


  cudaCheckError(cudaDeviceSynchronize());
  // And, we're all done.

}

