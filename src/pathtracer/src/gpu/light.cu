#include "light.h"
#include <iostream>

namespace VRRT {

__host__
SceneLight SceneLight::make(const SceneLight& light)
{
  SceneLight l;
  l.t = light.t;
  l.radiance = light.radiance;
  l.direction = light.direction;
  l.position = light.position;
  l.sampleToWorld = light.sampleToWorld;
  l.dim_x = light.dim_x;
  l.dim_y = light.dim_y;
  l.area = light.area;
  return l;
}


__host__
SceneLight SceneLight::make(const CMU462::Spectrum& rad,
                                   const CMU462::Vector3D& v, const LightType t)
{
  SceneLight l;
  l.radiance = Spectrum::make(rad);
  l.t = t;
  switch (t) {
    case DIRECTIONAL:
      l.direction = Vector3D::make(-v.unit());
      break;
    case POINT:
      l.position = Vector3D::make(v);
    default:
      std::cout << "Light Construction error\n";
  }
  return l;
}

__host__
SceneLight SceneLight::make(const CMU462::Spectrum& rad,
                                   const LightType t)
{
  SceneLight l;
  l.radiance = Spectrum::make(rad);
  l.t = t;
  CMU462::Matrix3x3 sampleToWorld;
  switch (t) {
    case HEMISPHERE:
      sampleToWorld[0] = CMU462::Vector3D(1,  0,  0);
      sampleToWorld[1] = CMU462::Vector3D(0,  0, -1);
      sampleToWorld[2] = CMU462::Vector3D(0,  1,  0);
      l.sampleToWorld = Matrix3x3::make(sampleToWorld);
      break;
    default:
      std::cout << "Light construction error\n";
  }
  return l;
}

__host__
SceneLight SceneLight::make(const CMU462::Spectrum& rad,
                                   const CMU462::Vector3D& pos,
                                   const CMU462::Vector3D& dir,
                                   const CMU462::Vector3D& dim_x,
                                   const CMU462::Vector3D& dim_y,
                                   const LightType t)
{
  SceneLight l;
  l.radiance = Spectrum::make(rad);
  l.position = Vector3D::make(pos);
  l.direction = Vector3D::make(dir);
  l.dim_x = Vector3D::make(dim_x);
  l.dim_y = Vector3D::make(dim_y);
  l.area = dim_x.norm() * dim_y.norm();
  l.t = t;
  if (t != AREA) std::cout << "Light Construction Error\n";
  return l;
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
      d = uniformHemisphereSample(state);
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
      sample = uniformGridSample(state) - Vector2D(0.5f, 0.5f);
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
