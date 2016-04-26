#include "bbox.h"

namespace VRRT {

__device__
bool BBox::intersect(Ray& r, double& t0, double& t1) const {

  // TODO:
  // Implement ray - bounding box intersection test
  // If the ray intersected the bouding box within the range given by
  // t0, t1, update t0 and t1 with the new intersection times.

  //get intersections with all 6 bounding planes
  double tx0 = (min.x - r.o.x) * r.inv_d.x;
  double tx1 = (max.x - r.o.x) * r.inv_d.x;
  double ty0 = (min.y - r.o.y) * r.inv_d.y;
  double ty1 = (max.y - r.o.y) * r.inv_d.y;
  double tz0 = (min.z - r.o.z) * r.inv_d.z;
  double tz1 = (max.z - r.o.z) * r.inv_d.z;
  //sort  intersection times for each dimension
  double txMin = (tx0 < tx1) ? tx0 : tx1;
  double txMax = (tx0 < tx1) ? tx1 : tx0;
  double tyMin = (ty0 < ty1) ? ty0 : ty1;
  double tyMax = (ty0 < ty1) ? ty1 : ty0;
  double tzMin = (tz0 < tz1) ? tz0 : tz1;
  double tzMax = (tz0 < tz1) ? tz1 : tz0;
  //get final intersect times
  double tMin = fmax(txMin, fmax(tyMin, tzMin));
  double tMax = fmin(txMax, fmin(tyMax, tzMax));

  if ((tMin <= tMax) && (tMin < t1) && (tMax > t0)) {
    t0 = fmax(tMin, t0);
    t1 = fmin(tMax, t1);
    //r.max_t = t1;
    return true;
  }
  return false;
}

} // namespace CMU462
