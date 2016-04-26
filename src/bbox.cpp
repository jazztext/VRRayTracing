#include "bbox.h"

#include "GL/glew.h"

#include <algorithm>
#include <iostream>

namespace CMU462 {

bool BBox::intersect(const Ray& r, double& t0, double& t1) const {

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
  double tMin = std::max(txMin, std::max(tyMin, tzMin));
  double tMax = std::min(txMax, std::min(tyMax, tzMax));

  if ((tMin <= tMax) && (tMin < t1) && (tMax > t0)) {
    t0 = std::max(tMin, t0);
    t1 = std::min(tMax, t1);
    //r.max_t = t1;
    return true;
  }
  return false;
}

void BBox::draw(Color c) const {

  glColor4f(c.r, c.g, c.b, c.a);

	// top
	glBegin(GL_LINE_STRIP);
	glVertex3d(max.x, max.y, max.z);
  glVertex3d(max.x, max.y, min.z);
  glVertex3d(min.x, max.y, min.z);
  glVertex3d(min.x, max.y, max.z);
  glVertex3d(max.x, max.y, max.z);
	glEnd();

	// bottom
	glBegin(GL_LINE_STRIP);
  glVertex3d(min.x, min.y, min.z);
  glVertex3d(min.x, min.y, max.z);
  glVertex3d(max.x, min.y, max.z);
  glVertex3d(max.x, min.y, min.z);
  glVertex3d(min.x, min.y, min.z);
	glEnd();

	// side
	glBegin(GL_LINES);
	glVertex3d(max.x, max.y, max.z);
  glVertex3d(max.x, min.y, max.z);
	glVertex3d(max.x, max.y, min.z);
  glVertex3d(max.x, min.y, min.z);
	glVertex3d(min.x, max.y, min.z);
  glVertex3d(min.x, min.y, min.z);
	glVertex3d(min.x, max.y, max.z);
  glVertex3d(min.x, min.y, max.z);
	glEnd();

}

std::ostream& operator<<(std::ostream& os, const BBox& b) {
  return os << "BBOX(" << b.min << ", " << b.max << ")";
}

} // namespace CMU462
