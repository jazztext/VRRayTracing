#ifndef __STATICSCENE_LIGHT_H__
#define __STATICSCENE_LIGHT_H__

#include "misc.h"
#include "sampler.h" // UniformHemisphereSampler3D, UniformGridSampler2D
//#include "../image.h"   // HDRImageBuffer

#include "vector3D.h"
#include "matrix3x3.h"
#include "spectrum.h"

namespace VRRT {

/**
 * Interface for lights in the scene.
 */
class SceneLight {
 public:
   enum LightType { DIRECTIONAL, HEMISPHERE, POINT, AREA };

   __host__
   SceneLight() { }
   __host__
   SceneLight(const SceneLight& light) :
     t(light.t), radiance(light.radiance), direction(light.direction),
     position(light.position), sampleToWorld(light.sampleToWorld),
     hemisphereSampler(light.hemisphereSampler), gridSampler(light.gridSampler),
     dim_x(light.dim_x), dim_y(light.dim_y), area(light.area) { }
   __host__
   SceneLight(const CMU462::Spectrum& rad, const CMU462::Vector3D& v,
              const LightType t);
   __host__
   SceneLight(const CMU462::Spectrum& rad, const LightType t);
   __host__
   SceneLight(const CMU462::Spectrum& rad, const CMU462::Vector3D& pos,
              const CMU462::Vector3D& dir, float angle, const LightType t);
   __host__
   SceneLight(const CMU462::Spectrum& rad,
              const CMU462::Vector3D& pos,   const CMU462::Vector3D& dir,
              const CMU462::Vector3D& dim_x, const CMU462::Vector3D& dim_y,
              const LightType t);

   __device__
   Spectrum sample_L(const Vector3D& p, Vector3D* wi, float* distToLight,
                     float* pdf, curandState *state) const;
   __device__
   bool is_delta_light() const;

 private:
   LightType t;
   Spectrum radiance;
   Vector3D direction;
   Vector3D position;
   Matrix3x3 sampleToWorld;
   UniformHemisphereSampler3D hemisphereSampler;
   UniformGridSampler2D gridSampler;
   Vector3D dim_x;
   Vector3D dim_y;
   float area;

};

} // namespace StaticScene

#endif  // CMU462_STATICSCENE_BSDF_H
