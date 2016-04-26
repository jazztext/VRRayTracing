#include "../camera.h"
#include "../static_scene/scene.h"
#include "../bvh.h"

namespace VRRT {

void raytrace_scene(CMU462::Camera *c, CMU462::StaticScene::Scene *scene,
                    CMU462::StaticScene::BVHAccel *bvh, int screenW,
                    int screenH, int ns_aa, int ns_area_light,
                    int max_num_rays);

}
