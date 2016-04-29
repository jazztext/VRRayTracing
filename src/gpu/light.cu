#include "light.h"
#include <iostream>

namespace VRRT {

// Directional Light //

__host__
SceneLight::SceneLight(const CMU462::Spectrum& rad, const CMU462::Vector3D& v,
                       const LightType t)
    : radiance(rad), t(t) {
  switch (t) {
    case DIRECTIONAL:
      direction = Vector3D(-v.unit());
      break;
    case POINT:
      position = v;
    default:
      std::cout << "Light Construction error\n";
  }
}

__host__
SceneLight::SceneLight(const CMU462::Spectrum& rad, const LightType t)
    : radiance(rad), t(t) {
  CMU462::Matrix3x3 sampleToWorldH;
  switch (t) {
    case HEMISPHERE:
      sampleToWorldH[0] = CMU462::Vector3D(1,  0,  0);
      sampleToWorldH[1] = CMU462::Vector3D(0,  0, -1);
      sampleToWorldH[2] = CMU462::Vector3D(0,  1,  0);
      sampleToWorld = Matrix3x3(sampleToWorldH);
      break;
    default:
      std::cout << "Light construction error\n";
  }
}

__host__
SceneLight::SceneLight(const CMU462::Spectrum& rad,
                       const CMU462::Vector3D& pos,
                       const CMU462::Vector3D& dir,
                       const CMU462::Vector3D& dim_x,
                       const CMU462::Vector3D& dim_y,
                       const LightType t)
  : radiance(rad), position(pos), direction(dir),
    dim_x(dim_x), dim_y(dim_y), area(dim_x.norm() * dim_y.norm()), t(t)
{
  if (t != AREA) std::cout << "Light Construction Error\n";
}

__device__
Spectrum SceneLight::sample_L(const Vector3D& p, Vector3D* wi,
                              float* distToLight, float* pdf,
                              curandState *state) const
{
  Vector3D d;
  Vector2D sample;
  float cosTheta, sqDist, dist;
  switch (t) {
    case DIRECTIONAL:
      *wi = direction;
      *distToLight = infy();
      *pdf = 1.0;
      return radiance;
    case HEMISPHERE:
      d = hemisphereSampler.get_sample(state);
      *wi = sampleToWorld* d;
      *distToLight = infy();
      *pdf = 1.0 / (2.0 * M_PI);
      return radiance;
    case POINT:
      d = position - p;
      *wi = d.unit();
      *distToLight = d.norm();
      *pdf = 1.0;
      return radiance;
    case AREA:
      sample = gridSampler.get_sample(state) - Vector2D(0.5f, 0.5f);
      d = position + sample.x * dim_x + sample.y * dim_y - p;
      cosTheta = dot(d, direction);
      sqDist = d.norm2();
      dist = sqrtf(sqDist);
      *wi = d / dist;
      *distToLight = dist - .01;
      *pdf = sqDist / (area * fabsf(cosTheta));
      return cosTheta < 0 ? radiance : Spectrum();
    default:
      return radiance;
  }
}

__device__
bool SceneLight::is_delta_light() const
{
  switch (t) {
    case DIRECTIONAL:
      return true;
    case HEMISPHERE:
      return false;
    case POINT:
      return true;
    case AREA:
      return false;
    default:
      return false;
  }
}

} // namespace VRRT
