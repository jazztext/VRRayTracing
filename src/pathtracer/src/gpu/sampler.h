#ifndef __SAMPLER_H__
#define __SAMPLER_H__

#include <curand_kernel.h>

#include "misc.h"
#include "vector2D.h"
#include "vector3D.h"

namespace VRRT {

/**
 * A Sampler2D implementation with uniform distribution on unit square
 */
__device__
Vector2D uniformGridSample(curandState *state);

/**
 * A Sampler3D implementation with uniform distribution on unit hemisphere
 */
__device__
Vector3D uniformHemisphereSample(curandState *state);

/**
 * A Sampler3D implementation with cosine-weighted distribution on unit
 * hemisphere.
 */
__device__
Vector3D cosineWeightedHemisphereSample(curandState *state);
// Also returns the pdf at the sample point for use in importance sampling.
__device__
Vector3D cosineWeightedHemisphereSample(curandState *state, float* pdf);

} // namespace CMU462

#endif //CMU462_SAMPLER_H
