#include "sampler.h"

namespace VRRT {

// Uniform Sampler2D Implementation //


__device__
Vector2D uniformGridSample(curandState *state) {
  return Vector2D(curand_uniform(state), curand_uniform(state));
}

// Uniform Hemisphere Sampler3D Implementation //
__device__
Vector3D uniformHemisphereSample(curandState *state) {

  float Xi1 = curand_uniform(state);
  float Xi2 = curand_uniform(state);

  float theta = acos(Xi1);
  float phi = 2.0 * PI * Xi2;

  float xs = sinf(theta) * cosf(phi);
  float ys = sinf(theta) * sinf(phi);
  float zs = cosf(theta);

  return Vector3D::make(xs, ys, zs);

}

__device__
Vector3D cosineWeightedHemisphereSample(curandState *state)
{
  float f;
  return cosineWeightedHemisphereSample(state, &f);
}

__device__
Vector3D cosineWeightedHemisphereSample(curandState *state, float *pdf) {
  float z1 = curand_uniform(state), z2 = curand_uniform(state);
  float theta = 2 * PI * z1, r = sqrtf(z2), z = sqrtf(1 - r*r);
  *pdf = z / PI;
  return Vector3D::make(r * cosf(theta), r * sinf(theta), z);
}


} // namespace CMU462
