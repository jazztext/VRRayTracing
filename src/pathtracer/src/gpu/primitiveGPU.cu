#include "primitiveGPU.h"

namespace VRRT {

PrimitiveGPU::PrimitiveGPU(CMU462::Vector3D p1, CMU462::Vector3D p2,
                           CMU462::Vector3D p3, CMU462::Vector3D n1,
                           CMU462::Vector3D n2, CMU462::Vector3D n3,
                           BSDF *bsdf)
{
  this->type = TRI;
  this->p1 = Vector3D(p1);
  this->p2 = Vector3D(p2);
  this->p3 = Vector3D(p3);
  this->n1 = Vector3D(n1);
  this->n2 = Vector3D(n2);
  this->n3 = Vector3D(n3);
  this->bsdf = bsdf;
}

PrimitiveGPU::PrimitiveGPU(CMU462::Vector3D o, float r, BSDF *bsdf)
{
  this->type = SPHERE;
  this->o = Vector3D(o);
  this->r = r;
  this->r2 = r * r;
  this->bsdf = bsdf;
}

__device__
BBox PrimitiveGPU::get_bbox()
{
  switch (type) {
    case TRI:
     return triangleGetBBox();
    case SPHERE:
     return sphereGetBBox();
  }
  return BBox(); // stop nvcc from complaining
}

__device__
bool PrimitiveGPU::intersect(Ray& r)
{
  Intersection i;
  switch (type) {
    case TRI:
      return triangleIntersect(r, &i);
    case SPHERE:
      return sphereIntersect(r, &i);
  }
  return false; // stop nvcc from complaining
}

__device__
bool PrimitiveGPU::intersect(Ray& r, Intersection* i)
{
  switch (type) {
    case TRI:
      return triangleIntersect(r, i);
    case SPHERE:
      return sphereIntersect(r, i);
  }
  return false; // stop nvcc from complaining
}

__device__
BBox PrimitiveGPU::triangleGetBBox()
{
  Vector3D min(fminf(p1.x, fminf(p2.x, p3.x)),
               fminf(p1.y, fminf(p2.y, p3.y)),
               fminf(p1.z, fminf(p2.z, p3.z)));

  Vector3D max(fmaxf(p1.x, fmaxf(p2.x, p3.x)),
               fmaxf(p1.y, fmaxf(p2.y, p3.y)),
               fmaxf(p1.z, fmaxf(p2.z, p3.z)));

  return BBox(min, max);
}

__device__
BBox PrimitiveGPU::sphereGetBBox()
{
  return BBox(o - Vector3D(r,r,r), o + Vector3D(r,r,r));
}

__device__
bool PrimitiveGPU::triangleIntersect(Ray& r, Intersection* i)
{
  Vector3D e1 = p2 - p1;
  Vector3D e2 = p3 - p1;
  Vector3D s = r.o - p1;
  Vector3D c1 = cross(e1, r.d), c2 = cross(e2, s);
  float denom = dot(c1, e2);
  float u = dot(c2, r.d) / denom, v = dot(c1, s) / denom;
  float t = dot(c2,e1) / denom;

  if (r.min_t <= t && t <= r.max_t && u >=0 && v >= 0 &&  (u + v) <= 1) {
    r.max_t = t;
    i->t = t;
    i->n = u * n2 + v * n3 +
           (1 - u - v) * n1;
    if (dot(i->n, r.d) > 0) i->n *= -1;
    i->primitive = this;
    i->bsdf = bsdf;
    return true;
  }
  return false;

}

__device__
bool PrimitiveGPU::test(Ray& r, float& t1, float& t2) const {
  Vector3D l = r.o - o;
  float disc = powf(dot(l, r.d), 2.) - dot(r.d, r.d) * (dot(l, l) - r2);
  if (disc < 0) return false;

  t1 = (-dot(l, r.d) - sqrtf(disc)) / dot(r.d, r.d);
  t2 = (-dot(l, r.d) + sqrtf(disc)) / dot(r.d, r.d);
  return true;
}


__device__
bool PrimitiveGPU::sphereIntersect(Ray& r, Intersection* i)
{
  float t1, t2;
  if (!test(r, t1, t2)) return false;
  if (r.min_t <= t1 && t1 <= r.max_t) {
    r.max_t = t1;
    i->t = t1;
    i->n = (r.o + r.d * t1 - o).unit();
    i->primitive = this;
    i->bsdf = bsdf;
    return true;
  }
  else if (r.min_t <= t2 && t2 <= r.max_t) {
    r.max_t = t2;
    i->t = t2;
    i->n = (r.o + r.d * t2 - o).unit();
    i->primitive = this;
    i->bsdf = bsdf;
    return true;
  }

  return false;
}

}
