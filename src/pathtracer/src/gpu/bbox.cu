#include "bbox.h"

namespace VRRT {

__device__
bool BBox::intersect(Ray& r, float& t0, float& t1) const {

  //get intersections with all 6 bounding planes
  float tx0 = (min.v.x - r.o.v.x) / r.d.v.x;
  float tx1 = (max.v.x - r.o.v.x) / r.d.v.x;
  float ty0 = (min.v.y - r.o.v.y) / r.d.v.y;
  float ty1 = (max.v.y - r.o.v.y) / r.d.v.y;
  float tz0 = (min.v.z - r.o.v.z) / r.d.v.z;
  float tz1 = (max.v.z - r.o.v.z) / r.d.v.z;
  //sort  intersection times for each dimension
  float txMin = (tx0 < tx1) ? tx0 : tx1;
  float txMax = (tx0 < tx1) ? tx1 : tx0;
  float tyMin = (ty0 < ty1) ? ty0 : ty1;
  float tyMax = (ty0 < ty1) ? ty1 : ty0;
  float tzMin = (tz0 < tz1) ? tz0 : tz1;
  float tzMax = (tz0 < tz1) ? tz1 : tz0;
  //get final intersect times
  float tMin = fmaxf(txMin, fmaxf(tyMin, tzMin));
  float tMax = fminf(txMax, fminf(tyMax, tzMax));

  if ((tMin <= tMax) && (tMin < t1) && (tMax > t0)) {
    t0 = fmaxf(tMin, t0);
    t1 = fminf(tMax, t1);
    return true;
  }
  return false;
}

} // namespace CMU462
