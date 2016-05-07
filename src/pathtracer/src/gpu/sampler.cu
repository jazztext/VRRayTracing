#include "sampler.h"

namespace VRRT {

// Uniform Sampler2D Implementation //


__device__
Vector2D UniformGridSampler2D::get_sample(curandState *state) const {

  // TODO:
  // Implement uniform 2D grid sampler
  return Vector2D(curand_uniform(state), curand_uniform(state));
}

// Uniform Hemisphere Sampler3D Implementation //
__device__
Vector3D UniformHemisphereSampler3D::get_sample(curandState *state) const {

  float Xi1 = curand_uniform(state);
  float Xi2 = curand_uniform(state);

  float theta = acos(Xi1);
  float phi = 2.0 * PI * Xi2;

  float xs = sinf(theta) * cosf(phi);
  float ys = sinf(theta) * sinf(phi);
  float zs = cosf(theta);

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
  float theta = 2 * PI * z1, r = sqrtf(z2), z = sqrtf(1 - r*r);
  *pdf = z / PI;
  return Vector3D(r * cosf(theta), r * sinf(theta), z);
}


} // namespace CMU462
