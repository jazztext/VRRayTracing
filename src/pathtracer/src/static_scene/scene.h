#ifndef CMU462_STATICSCENE_SCENE_H
#define CMU462_STATICSCENE_SCENE_H

#include "CMU462/CMU462.h"
#include "primitive.h"
#include "../gpu/light.h"

#include <vector>

namespace CMU462 { namespace StaticScene {

/**
 * Interface for objects in the scene.
 */
class SceneObject {
 public:

  /**
   * Get all the primitives in the scene object.
   * \return a vector of all the primitives in the scene object
   */
  virtual std::vector<Primitive*> get_primitives() const = 0;

  /**
   * Get the surface BSDF of the object's surface.
   * \return the BSDF of the objects's surface
   */
  virtual VRRT::BSDF* get_bsdf() const = 0;

  virtual BBox get_bbox() = 0;

};



/**
 * Represents a scene in a raytracer-friendly format. To speed up raytracing,
 * all data is already transformed to world space.
 */
struct Scene {
  Scene(const std::vector<SceneObject *>& objects,
        const std::vector<VRRT::SceneLight>& lights)
    : objects(objects), lights(lights) { }

  // kept to make sure they don't get deleted, in case the
  //  primitives depend on them (e.g. Mesh Triangles).
  std::vector<SceneObject*> objects;

  // for sake of consistency of the scene object Interface
  std::vector<VRRT::SceneLight> lights;
 

  // TODO (sky) :
  // Adding object with emission BSDFs as mesh lights and sphere lights so
  // that light sampling configurations also applies to mesh lights.
//  for (SceneObject *obj : objects) {
//    if (obj->get_material().emit != Spectrum()) {
//
//      // mesh light
//      if (dynamic_cast<Mesh*>(obj)) {
//        staticLights.push_back()
//      }
//
//      // sphere light
//      if (dynamic_cast<Sphere*>(obj)) {
//
//      }
//  }

};

} // namespace StaticScene
} // namespace CMU462

#endif //CMU462_STATICSCENE_SCENE_H
