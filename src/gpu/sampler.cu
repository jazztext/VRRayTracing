#include "sampler.h"

namespace VRRT {

// Uniform Sampler2D Implementation //


__device__
Vector2D UniformGridSampler2D::get_sample(curandState *state) const {

  // TODO:
  // Implement uniform 2D grid sampler
  return Vector2D(curand_uniform_double(state), curand_uniform_double(state));
}

// Uniform Hemisphere Sampler3D Implementation //
__device__
Vector3D UniformHemisphereSampler3D::get_sample(curandState *state) const {

  double Xi1 = curand_uniform_double(state);
  double Xi2 = curand_uniform_double(state);

  double theta = acos(Xi1);
  double phi = 2.0 * PI * Xi2;

  double xs = sinf(theta) * cosf(phi);
  double ys = sinf(theta) * sinf(phi);
  double zs = cosf(theta);

  return Vector3D(xs, ys, zs);

}

__device__
Vector3D CosineWeightedHemisphereSampler3D::get_sample(curandState *state) const
{
  float f;
  return get_sample(state, &f);
}

__device__
Vector3D CosineWeightedHemisphereSampler3D::get_sample(curandState *state,
                                                       float *pdf) const {
  // You may implement this, but don't have to.
  float z1 = curand_uniform(state), z2 = curand_uniform(state);
  float theta = 2 * PI * z1, r = sqrt(z2), z = sqrt(1 - r*r);
  *pdf = z / PI;
  return Vector3D(r * cos(theta), r * sin(theta), z);
}


} // namespace CMU462
