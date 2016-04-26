#include "application.h"
#include "gpu/pathtracer.h"

using Collada::CameraInfo;
using Collada::LightInfo;
using Collada::MaterialInfo;
using Collada::PolymeshInfo;
using Collada::SceneInfo;
using Collada::SphereInfo;

namespace CMU462 {

Application::Application(AppConfig config) {
  ns_aa = config.pathtracer_ns_aa;
  ns_area_light = config.pathtracer_ns_area_light;
  max_ray_depth = config.pathtracer_max_ray_depth;
  init();
}

Application::~Application() {

}

void Application::init() {

  scene = nullptr;

  // Make a dummy camera so resize() doesn't crash before the scene has been
  // loaded.
  // NOTE: there's a chicken-and-egg problem here, because loadScene
  // requires init, and init requires init_camera (which is only called by
  // loadScene).
  screenW = 960;
  screenH = 640; // Default value
  CameraInfo cameraInfo;
  cameraInfo.hFov = 50;
  cameraInfo.vFov = 35;
  cameraInfo.nClip = 0.01;
  cameraInfo.fClip = 100;
  camera.configure(cameraInfo, screenW, screenH);
}

void Application::resize(size_t w, size_t h) {
  screenW = w;
  screenH = h;
  camera.set_screen_size(w, h);
}

string Application::name() {
  return "PathTracer";
}

string Application::info() {
  return "PathTracer";
}

void Application::load(SceneInfo* sceneInfo) {

  vector<Collada::Node>& nodes = sceneInfo->nodes;
  vector<VRRT::SceneLight *> lights;
  vector<StaticScene::SceneObject *> objects;

  // save camera position to update camera control later
  CameraInfo *c;
  Vector3D c_pos = Vector3D();
  Vector3D c_dir = Vector3D();

  int len = nodes.size();
  for (int i = 0; i < len; i++) {
    Collada::Node& node = nodes[i];
    Collada::Instance *instance = node.instance;
    const Matrix4x4& transform = node.transform;

    switch(instance->type) {
      case Collada::Instance::CAMERA:
        c = static_cast<CameraInfo*>(instance);
        c_pos = (transform * Vector4D(c_pos,1)).to3D();
        c_dir = (transform * Vector4D(c->view_dir,1)).to3D().unit();
        init_camera(*c, transform);
        break;
      case Collada::Instance::LIGHT:
      {
        lights.push_back(
          init_light(static_cast<LightInfo&>(*instance), transform));
        break;
      }
      case Collada::Instance::SPHERE:
        objects.push_back(
          init_sphere(static_cast<SphereInfo&>(*instance), transform));
        break;
      case Collada::Instance::POLYMESH:
        objects.push_back(
          init_polymesh(static_cast<PolymeshInfo&>(*instance), transform));
        break;
      case Collada::Instance::MATERIAL:
        init_material(static_cast<MaterialInfo&>(*instance));
        break;
     }
  }

  scene = new StaticScene::Scene(objects, lights);

  BBox bbox;
  for (StaticScene::SceneObject *obj : objects) {
    bbox.expand(obj->get_bbox());
  }
  if (!bbox.empty()) {

    Vector3D target = bbox.centroid();
    canonical_view_distance = bbox.extent.norm() / 2 * 1.5;

    double view_distance = canonical_view_distance * 2;
    double min_view_distance = canonical_view_distance / 10.0;
    double max_view_distance = canonical_view_distance * 20.0;


    canonicalCamera.place(target,
                          acos(c_dir.y),
                          atan2(c_dir.x, c_dir.z),
                          view_distance,
                          min_view_distance,
                          max_view_distance);

    camera.place(target,
                acos(c_dir.y),
                atan2(c_dir.x, c_dir.z),
                view_distance,
                min_view_distance,
                max_view_distance);

    std::cout << camera.position() << " " << camera.view_point() << "\n";

  }

  // set default draw styles for meshEdit -

}

void Application::init_camera(CameraInfo& cameraInfo,
                              const Matrix4x4& transform) {
  camera.configure(cameraInfo, screenW, screenH);
  canonicalCamera.configure(cameraInfo, screenW, screenH);
}

void Application::reset_camera() {
  camera.copy_placement(canonicalCamera);
}

VRRT::SceneLight *Application::init_light(LightInfo& light,
                                                 const Matrix4x4& transform) {
  Vector3D position, direction, dim_x, dim_y, dim_xT, dim_yT;
  switch(light.light_type) {
    case Collada::LightType::NONE:
      break;
    case Collada::LightType::AMBIENT:
      return new VRRT::InfiniteHemisphereLight(light.spectrum);
    case Collada::LightType::DIRECTIONAL:
      direction = -(transform * Vector4D(light.direction, 1)).to3D();
      direction.normalize();
      return new VRRT::DirectionalLight(light.spectrum, direction);
    case Collada::LightType::AREA:
      position = (transform * Vector4D(light.position, 1)).to3D();
      direction = (transform * Vector4D(light.direction, 1)).to3D() - position;
      direction.normalize();

      dim_y = light.up;
      dim_x = cross(light.up, light.direction);

      dim_xT = (transform * Vector4D(dim_x, 1)).to3D() - position;
      dim_yT = (transform * Vector4D(dim_y, 1)).to3D() - position;

      return new VRRT::AreaLight(light.spectrum, position, direction,
                                 dim_xT, dim_yT);
    case Collada::LightType::POINT:
      position = (transform * Vector4D(light.position, 1)).to3D();
      return new VRRT::PointLight(light.spectrum, direction);
    case Collada::LightType::SPOT:
      position = (transform * Vector4D(light.position, 1)).to3D();
      direction = (transform * Vector4D(light.direction, 1)).to3D() - position;
      direction.normalize();
      return new VRRT::SpotLight(light.spectrum, position, direction, PI * .5f);
    default:
      break;
  }
  return nullptr;
}

/**
 * The transform is assumed to be composed of translation, rotation, and
 * scaling, where the scaling is uniform across the three dimensions; these
 * assumptions are necessary to ensure the sphere is still spherical. Rotation
 * is ignored since it's a sphere, translation is determined by transforming the
 * origin, and scaling is determined by transforming an arbitrary unit vector.
 */
StaticScene::SceneObject *Application::init_sphere(SphereInfo& sphere,
                                                   const Matrix4x4& transform) {
  const Vector3D& position = (transform * Vector4D(0, 0, 0, 1)).projectTo3D();
  double scale = (transform * Vector4D(1, 0, 0, 0)).to3D().norm();
  VRRT::BSDF *bsdf;
  if (sphere.material) {
    bsdf = sphere.material->bsdf;
  } else {
    bsdf = new VRRT::DiffuseBSDF(Spectrum(0.5f,0.5f,0.5f));
  }
  return new StaticScene::SphereObject(position, sphere.radius * scale, bsdf);
}

StaticScene::SceneObject *Application::init_polymesh(PolymeshInfo& polymesh,
                                               const Matrix4x4& transform) {
  vector< vector<size_t> > polygons;
  for (const Collada::Polygon& p : polymesh.polygons) {
    polygons.push_back(p.vertex_indices);
  }
  vector<Vector3D> vertices = polymesh.vertices; // DELIBERATE COPY.
  for (int i = 0; i < vertices.size(); i++) {
    vertices[i] = (transform * Vector4D(vertices[i], 1)).projectTo3D();
  }

  HalfedgeMesh mesh;
  VRRT::BSDF *bsdf;
  mesh.build(polygons, vertices);
  if (polymesh.material) {
    bsdf = polymesh.material->bsdf;
  } else {
    bsdf = new VRRT::DiffuseBSDF(Spectrum(0.5f,0.5f,0.5f));
  }

  return new StaticScene::Mesh(mesh, bsdf);
}

void Application::init_material(MaterialInfo& material) {
  // TODO : Support Materials.
}

void Application::pathtrace() {
  vector<StaticScene::Primitive *> primitives;
  for (StaticScene::SceneObject *obj : scene->objects) {
    const vector<StaticScene::Primitive *> &obj_prims = obj->get_primitives();
    primitives.reserve(primitives.size() + obj_prims.size());
    primitives.insert(primitives.end(), obj_prims.begin(), obj_prims.end());
  }
  StaticScene::BVHAccel *bvh = new StaticScene::BVHAccel(primitives, 4);

  VRRT::raytrace_scene(&camera, scene, bvh, screenW, screenH, ns_aa,
                       ns_area_light, max_ray_depth);

}

} // namespace CMU462
