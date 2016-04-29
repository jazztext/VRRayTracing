#ifndef CMU462_APPLICATION_H
#define CMU462_APPLICATION_H

// STL
#include <string>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <string>
#include <vector>

// libCMU462
#include "CMU462/CMU462.h"

// COLLADA
#include "collada/collada.h"
#include "collada/light_info.h"
#include "collada/sphere_info.h"
#include "collada/polymesh_info.h"
#include "collada/material_info.h"

// MeshEdit
#include "halfEdgeMesh.h"
#include "bvh.h"

// PathTracer
#include "static_scene/scene.h"
#include "static_scene/object.h"
//#include "image.h"

// Shared modules
#include "camera.h"

using namespace std;

namespace CMU462 {

struct AppConfig {

  AppConfig () {

    pathtracer_ns_aa = 1;
    pathtracer_max_ray_depth = 1;
    pathtracer_ns_area_light = 4;

    pathtracer_ns_diff = 1;
    pathtracer_ns_glsy = 1;
    pathtracer_ns_refr = 1;

    pathtracer_num_threads = 1;

  }

  size_t pathtracer_ns_aa;
  size_t pathtracer_max_ray_depth;
  size_t pathtracer_ns_area_light;
  size_t pathtracer_ns_diff;
  size_t pathtracer_ns_glsy;
  size_t pathtracer_ns_refr;
  size_t pathtracer_num_threads;

};

class Application {
 public:

  Application(AppConfig config);

  ~Application();

  void init();
  void resize(size_t w, size_t h);

  std::string name();
  std::string info();

  void load(Collada::SceneInfo* sceneInfo);
  void pathtrace();

 private:

  void set_up_pathtracer();

  StaticScene::Scene *scene;

  // View Frustrum Variables.
  // On resize, the aspect ratio is changed. On reset_camera, the position and
  // orientation are reset but NOT the aspect ratio.
  Camera camera;
  Camera canonicalCamera;

  size_t screenW;
  size_t screenH;
  int ns_aa, ns_area_light, max_ray_depth;

  // Length of diagonal of bounding box for the mesh.
  // Guranteed to not have the camera occlude with the mes.
  double canonical_view_distance;

  /**
   * Reads and combines the current modelview and projection matrices.
   */
  Matrix4x4 get_world_to_3DH();

  void init_camera(Collada::CameraInfo& camera, const Matrix4x4& transform);
  VRRT::SceneLight init_light(Collada::LightInfo& light, const Matrix4x4& transform);
  StaticScene::SceneObject *init_sphere(Collada::SphereInfo& polymesh, const Matrix4x4& transform);
  StaticScene::SceneObject *init_polymesh(Collada::PolymeshInfo& polymesh, const Matrix4x4& transform);
  void init_material(Collada::MaterialInfo& material);

  // Resets the camera to the canonical initial view position.
  void reset_camera();

}; // class Application

} // namespace CMU462

  #endif // CMU462_APPLICATION_H
