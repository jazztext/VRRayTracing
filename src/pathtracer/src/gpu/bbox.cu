#include "bbox.h"

namespace VRRT {

__device__
bool BBox::intersect(Ray& r, float& t0, float& t1) const {

  //get intersections with all 6 bounding planes
  float tx0 = (min.x - r.o.x) * r.inv_d.x;
  float tx1 = (max.x - r.o.x) * r.inv_d.x;
  float ty0 = (min.y - r.o.y) * r.inv_d.y;
  float ty1 = (max.y - r.o.y) * r.inv_d.y;
  float tz0 = (min.z - r.o.z) * r.inv_d.z;
  float tz1 = (max.z - r.o.z) * r.inv_d.z;
  //sort  intersection times for each dimension
  float txMin = (tx0 < tx1) ? tx0 : tx1;
  float txMax = (tx0 < tx1) ? tx1 : tx0;
  float tyMin = (ty0 < ty1) ? ty0 : ty1;
  float tyMax = (ty0 < ty1) ? ty1 : ty0;
  float tzMin = (tz0 < tz1) ? tz0 : tz1;
  float tzMax = (tz0 < tz1) ? tz1 : tz0;
  //get final intersect times
  float tMin = fmax(txMin, fmax(tyMin, tzMin));
  float tMax = fmin(txMax, fmin(tyMax, tzMax));

  if ((tMin <= tMax) && (tMin < t1) && (tMax > t0)) {
    t0 = fmax(tMin, t0);
    t1 = fmin(tMax, t1);
    //r.max_t = t1;
    return true;
  }
  return false;
}

} // namespace CMU462
