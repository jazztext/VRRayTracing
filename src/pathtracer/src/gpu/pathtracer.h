#include "../camera.h"
#include "../static_scene/scene.h"
#include "../bvh.h"

namespace VRRT {

struct constantParams {
  int numLights, max_ray_depth, ns_aa, ns_area_light, w, h;
  Vector3D *points, *normals;
};

void raytrace_scene(CMU462::Camera *c, CMU462::StaticScene::Scene *scene,
                    CMU462::StaticScene::BVHAccel *bvh, int screenW,
                    int screenH, int ns_aa, int ns_area_light,
                    int max_num_rays, char *fname);

}
