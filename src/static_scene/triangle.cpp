#include "triangle.h"

#include "CMU462/CMU462.h"
#include "GL/glew.h"

namespace CMU462 { namespace StaticScene {

Triangle::Triangle(const Mesh* mesh, size_t v1, size_t v2, size_t v3) :
    mesh(mesh), v1(v1), v2(v2), v3(v3) { }

BBox Triangle::get_bbox() const {

  // TODO:
  // Compute the bounding box of the triangle.
  Vector3D p1 = mesh->positions[v1];
  Vector3D p2 = mesh->positions[v2];
  Vector3D p3 = mesh->positions[v3];

  Vector3D min(std::min(p1.x, std::min(p2.x, p3.x)),
               std::min(p1.y, std::min(p2.y, p3.y)),
               std::min(p1.z, std::min(p2.z, p3.z)));

  Vector3D max(std::max(p1.x, std::max(p2.x, p3.x)),
               std::max(p1.y, std::max(p2.y, p3.y)),
               std::max(p1.z, std::max(p2.z, p3.z)));

  return BBox(min, max);

}

bool Triangle::intersect(const Ray& r) const {

  // TODO:
  // Implement ray - triangle intersection.
  Vector3D e1 = mesh->positions[v2] - mesh->positions[v1];
  Vector3D e2 = mesh->positions[v3] - mesh->positions[v1];
  Vector3D s = r.o - mesh->positions[v1];
  Vector3D c1 = cross(e1, r.d), c2 = cross(e2, s);
  double denom = dot(c1, e2);
  double u = dot(c2, r.d) / denom, v = dot(c1, s) / denom;
  double t = dot(c2,e1) / denom;

  return (r.min_t <= t && t <= r.max_t && u + v < 1);

}

bool Triangle::intersect(const Ray& r, Intersection *i) const {

  // TODO:
  // Implement ray - triangle intersection.
  // When an intersection takes place, the Intersection data should
  // be updated correspondingly.

  Vector3D e1 = mesh->positions[v2] - mesh->positions[v1];
  Vector3D e2 = mesh->positions[v3] - mesh->positions[v1];
  Vector3D s = r.o - mesh->positions[v1];
  Vector3D c1 = cross(e1, r.d), c2 = cross(e2, s);
  double denom = dot(c1, e2);
  double u = dot(c2, r.d) / denom, v = dot(c1, s) / denom;
  double t = dot(c2,e1) / denom;

  if (r.min_t <= t && t <= r.max_t && u >=0 && v >= 0 &&  (u + v) <= 1) {
    r.max_t = t;
    i->t = t;
    i->n = u * mesh->normals[v2] + v * mesh->normals[v3] +
           (1 - u - v) * mesh->normals[v1];
    if (dot(i->n, r.d) > 0) i->n *= -1;
    i->primitive = this;
    i->bsdf = get_bsdf();
    return true;
  }
  return false;

}

VRRT::PrimitiveGPU Triangle::toGPU(std::unordered_map<VRRT::BSDF *, VRRT::BSDF *> &bsdfs)
{
  Vector3D p1 = mesh->positions[v1];
  Vector3D p2 = mesh->positions[v2];
  Vector3D p3 = mesh->positions[v3];
  Vector3D n1 = mesh->normals[v1];
  Vector3D n2 = mesh->normals[v2];
  Vector3D n3 = mesh->normals[v3];

  if (!bsdfs.count(get_bsdf())) {
    bsdfs.emplace(get_bsdf(), get_bsdf()->copyToDev());
  }

  return VRRT::PrimitiveGPU(p1, p2, p3, n1, n2, n3, bsdfs[get_bsdf()]);
}

} // namespace StaticScene
} // namespace CMU462
