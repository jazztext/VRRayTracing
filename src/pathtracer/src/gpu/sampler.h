#ifndef __SAMPLER_H__
#define __SAMPLER_H__

#include <curand_kernel.h>

#include "misc.h"
#include "vector2D.h"
#include "vector3D.h"

namespace VRRT {

/**
 * Interface for generating point samples within the unit square
 */
class Sampler2D {
 public:

  /**
   * Virtual destructor.
   */
  __device__
  virtual ~Sampler2D() { }

  /**
   * Take a point sample of the unit square
   */
  __device__
  virtual Vector2D get_sample(curandState *state) const = 0;

}; // class Sampler2D

/**
 * Interface for generating 3D vector samples
 */
class Sampler3D {
 public:

  /**
   * Virtual destructor.
   */
  __device__
  virtual ~Sampler3D() { }

  /**
   * Take a vector sample of the unit hemisphere
   */
  __device__
  virtual Vector3D get_sample(curandState *state) const = 0;

}; // class Sampler3D


/**
 * A Sampler2D implementation with uniform distribution on unit square
 */
class UniformGridSampler2D : public Sampler2D {
 public:

  __device__
  Vector2D get_sample(curandState *state) const;

}; // class UniformSampler2D

/**
 * A Sampler3D implementation with uniform distribution on unit hemisphere
 */
class UniformHemisphereSampler3D : public Sampler3D {
 public:

  __device__
  Vector3D get_sample(curandState *state) const;

}; // class UniformHemisphereSampler3D

/**
 * A Sampler3D implementation with cosine-weighted distribution on unit
 * hemisphere.
 */
class CosineWeightedHemisphereSampler3D : public Sampler3D {
 public:

  __device__
  Vector3D get_sample(curandState *state) const;
  // Also returns the pdf at the sample point for use in importance sampling.
  __device__
  Vector3D get_sample(curandState *state, float* pdf) const;

}; // class UniformHemisphereSampler3D

/**
 * TODO (extra credit) :
 * Jittered sampler implementations
 */

} // namespace CMU462

#endif //CMU462_SAMPLER_H
