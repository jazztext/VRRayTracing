#ifndef __PATHTRACER__
#define __PATHTRACER__

#include "../camera.h"
#include "../static_scene/scene.h"
#include "../bvh.h"
#include "bvhGPU.h"

namespace VRRT {

struct constantParams {
  int numLights, max_ray_depth, ns_aa, ns_area_light, w, h;
  Vector3D *points, *normals;
};

void copyTocuGlobals(struct constantParams *params);
void copyTocuLights(SceneLight *data, size_t size);

__global__ void initCurand(curandState *state);

__global__ void initBuffer(unsigned char *outBuffer);

__global__ void raytrace_pixel(unsigned char *img, CMU462::Camera c, BVHGPU bvh,
                               curandState *state);

void raytrace_scene(CMU462::Camera *c, CMU462::StaticScene::Scene *scene,
                    CMU462::StaticScene::BVHAccel *bvh, int screenW,
                    int screenH, int ns_aa, int ns_area_light,
                    int max_num_rays, char *fname);

}

#endif
