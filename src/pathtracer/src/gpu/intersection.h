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

  float t;    ///< time of intersection

  const PrimitiveGPU* primitive;  ///< the primitive intersected

  Vector3D n;  ///< normal at point of intersection

  BSDF* bsdf; ///< BSDF of the surface at point of intersection

};

} // namespace CMU462

#endif // CMU462_INTERSECT_H
