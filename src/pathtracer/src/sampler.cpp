#include "sampler.h"

namespace CMU462 {

// Uniform Sampler2D Implementation //

Vector2D UniformGridSampler2D::get_sample() const {

  // TODO:
  // Implement uniform 2D grid sampler
  return Vector2D((double) std::rand() / RAND_MAX,
                  (double) std::rand() / RAND_MAX);
}

// Uniform Hemisphere Sampler3D Implementation //

Vector3D UniformHemisphereSampler3D::get_sample() const {

  double Xi1 = (double)(std::rand()) / RAND_MAX;
  double Xi2 = (double)(std::rand()) / RAND_MAX;

  double theta = acos(Xi1);
  double phi = 2.0 * PI * Xi2;

  double xs = sinf(theta) * cosf(phi);
  double ys = sinf(theta) * sinf(phi);
  double zs = cosf(theta);

  return Vector3D(xs, ys, zs);

}

Vector3D CosineWeightedHemisphereSampler3D::get_sample() const {
  float f;
  return get_sample(&f);
}

Vector3D CosineWeightedHemisphereSampler3D::get_sample(float *pdf) const {
  // You may implement this, but don't have to.
  static bool seeded = false;
  if (!seeded) {
    seeded = true;
    srand(time(NULL));
  }
  float z1 = (float) rand() / RAND_MAX, z2 = (float) rand() / RAND_MAX;
  float theta = 2 * PI * z1, r = sqrt(z2), z = sqrt(1 - pow(r, 2));
  *pdf = z / PI;
  return Vector3D(r * cos(theta), r * sin(theta), z);
}


} // namespace CMU462
