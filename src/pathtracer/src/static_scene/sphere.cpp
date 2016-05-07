#include "sphere.h"

#include <cmath>

#include  "../gpu/bsdf.h"

namespace CMU462 { namespace StaticScene {

bool Sphere::test(const Ray& r, double& t1, double& t2) const {

  // TODO:
  // Implement ray - sphere intersection test.
  // Return true if there are intersections and writing the
  // smaller of the two intersection times in t1 and the larger in t2.

  Vector3D l = r.o - o;
  double disc= pow(dot(l, r.d), 2) - dot(r.d, r.d) * (dot(l, l) - r2);
  if (disc < 0) return false;

  t1 = (-dot(l, r.d) - sqrt(disc)) / dot(r.d, r.d);
  t2 = (-dot(l, r.d) + sqrt(disc)) / dot(r.d, r.d);
  return true;

}

bool Sphere::intersect(const Ray& r) const {

  // TODO:
  // Implement ray - sphere intersection.
  // Note that you might want to use the the Sphere::test helper here.

  double t1, t2;
  if (test(r, t1, t2) && ((r.min_t <= t1 && t1 <= r.max_t) ||
                          (r.min_t <= t2 && t2 <= r.max_t))) return true;
  else return false;

}

bool Sphere::intersect(const Ray& r, Intersection *i) const {

  // TODO:
  // Implement ray - sphere intersection.
  // Note again that you might want to use the the Sphere::test helper here.
  // When an intersection takes place, the Intersection data should be updated
  // correspondingly.

  double t1, t2;
  if (!test(r, t1, t2)) return false;
  if (r.min_t <= t1 && t1 <= r.max_t) {
    r.max_t = t1;
    i->t = t1;
    i->n = (r.o + r.d * t1 - o).unit();
    i->primitive = this;
    i->bsdf = get_bsdf();
    return true;
  }
  else if (r.min_t <= t2 && t2 <= r.max_t) {
    r.max_t = t2;
    i->t = t2;
    i->n = (r.o + r.d * t2 - o).unit();
    i->primitive = this;
    i->bsdf = get_bsdf();
    return true;
  }

  return false;

}

VRRT::PrimitiveGPU Sphere::toGPU(std::unordered_map<VRRT::BSDF *, VRRT::BSDF *> &bsdfs, std::unordered_map<const Mesh *, int> &meshes, int &nextOff)
{
  if (!bsdfs.count(get_bsdf())) {
    bsdfs.emplace(get_bsdf(), get_bsdf()->copyToDev());
  }
  return VRRT::PrimitiveGPU(o, r, bsdfs[get_bsdf()]);
}

} // namespace StaticScene
} // namespace CMU462
