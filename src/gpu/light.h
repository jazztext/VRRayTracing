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
  __device__
  virtual Spectrum sample_L(const Vector3D& p, Vector3D* wi,
                            float* distToLight, float* pdf,
                            curandState *state) const = 0;
  __device__
  virtual bool is_delta_light() const = 0;

  __host__
  virtual void copyToDev(SceneLight **dest) = 0;

};

// Directional Light //

class DirectionalLight : public SceneLight {
 public:
  __host__
  DirectionalLight(const CMU462::Spectrum& rad,
                   const CMU462::Vector3D& lightDir);
  __device__
  DirectionalLight(const Spectrum& rad, const Vector3D& lightDir);
  __device__
  Spectrum sample_L(const Vector3D& p, Vector3D* wi, float* distToLight,
                    float* pdf, curandState *state) const;
  __device__
  bool is_delta_light() const { return true; }
  __host__
  void copyToDev(SceneLight **dest);

 private:
  Spectrum radiance;
  Vector3D dirToLight;

}; // class Directional Light

// Infinite Hemisphere Light //

class InfiniteHemisphereLight : public SceneLight {
 public:
  __host__
  InfiniteHemisphereLight(const CMU462::Spectrum& rad);
  __device__
  InfiniteHemisphereLight(const Spectrum& rad);
  __device__
  Spectrum sample_L(const Vector3D& p, Vector3D* wi, float* distToLight,
                    float* pdf, curandState *state) const;
  __device__
  bool is_delta_light() const { return false; }
  void copyToDev(SceneLight **dest);


 private:
  Spectrum radiance;
  Matrix3x3 sampleToWorld;
  UniformHemisphereSampler3D sampler;

}; // class InfiniteHemisphereLight


// Point Light //

class PointLight : public SceneLight {
 public:
  __host__
  PointLight(const CMU462::Spectrum& rad, const CMU462::Vector3D& pos);
  __device__
  PointLight(const Spectrum& rad, const Vector3D& pos);
  __device__
  Spectrum sample_L(const Vector3D& p, Vector3D* wi, float* distToLight,
                    float* pdf, curandState *state) const;
  __device__
  bool is_delta_light() const { return true; }
  void copyToDev(SceneLight **dest);

 private:
  Spectrum radiance;
  Vector3D position;

}; // class PointLight

// Spot Light //

class SpotLight : public SceneLight {
 public:
  __host__
  SpotLight(const CMU462::Spectrum& rad, const CMU462::Vector3D& pos,
            const CMU462::Vector3D& dir, float angle);
  __device__
  SpotLight(const Spectrum& rad, const Vector3D& pos,
            const Vector3D& dir, float angle);

  __device__
  Spectrum sample_L(const Vector3D& p, Vector3D* wi, float* distToLight,
                    float* pdf, curandState *state) const;
  __device__
  bool is_delta_light() const { return true; }
  void copyToDev(SceneLight **dest);

 private:
  Spectrum radiance;
  Vector3D position;
  Vector3D direction;
  float angle;

}; // class SpotLight

// Area Light //

class AreaLight : public SceneLight {
 public:
  __host__
  AreaLight(const CMU462::Spectrum& rad,
            const CMU462::Vector3D& pos,   const CMU462::Vector3D& dir,
            const CMU462::Vector3D& dim_x, const CMU462::Vector3D& dim_y);
  __device__
  AreaLight(const Spectrum& rad,
            const Vector3D& pos,   const Vector3D& dir,
            const Vector3D& dim_x, const Vector3D& dim_y);
  __device__
  Spectrum sample_L(const Vector3D& p, Vector3D* wi, float* distToLight,
                    float* pdf, curandState *state) const;
  __device__
  bool is_delta_light() const { return false; }
  void copyToDev(SceneLight **dest);

 private:
  Spectrum radiance;
  Vector3D position;
  Vector3D direction;
  Vector3D dim_x;
  Vector3D dim_y;
  UniformGridSampler2D sampler;
  float area;

}; // class AreaLight

} // namespace StaticScene

#endif  // CMU462_STATICSCENE_BSDF_H
