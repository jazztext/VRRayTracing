#ifndef __INTERSECT_H__
#define __INTERSECT_H__

#include "vector3D.h"
#include "spectrum.h"
#include "misc.h"
#include "bsdf.h"

namespace VRRT {

class PrimitiveGPU;

/**
 * A record of an intersection point which includes the time of intersection
 * and other information needed for shading
 */
struct Intersection {

  __device__
  Intersection() : t (infy()), primitive(NULL), bsdf(NULL) { }

  Vector3D n;  ///< normal at point of intersection

  float t;    ///< time of intersection

  const PrimitiveGPU* primitive;  ///< the primitive intersected

  BSDF* bsdf; ///< BSDF of the surface at point of intersection

};

struct IntersectionSoA {
  Vector3D n[256];
  float t[256];
  const PrimitiveGPU* primitive[256];
  BSDF *bsdf[256];
};

} // namespace CMU462

#endif // CMU462_INTERSECT_H
