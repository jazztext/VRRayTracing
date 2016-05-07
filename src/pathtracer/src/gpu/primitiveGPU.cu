#include "primitiveGPU.h"
#include "pathtracer.h"

namespace VRRT {

extern __constant__ constantParams cuGlobals;

PrimitiveGPU::PrimitiveGPU(int i1, int i2, int i3, BSDF *bsdf)
{
  this->type = TRI;
  this->inds = make_int3(i1, i2, i3);
  this->bsdf = bsdf;
}

PrimitiveGPU::PrimitiveGPU(CMU462::Vector3D o, float r, BSDF *bsdf)
{
  this->type = SPHERE;
  this->o = Vector3D::make(o);
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
    default:
     return BBox();
  }
}

__device__
bool PrimitiveGPU::intersect(Ray& r)
{
  IntersectionSoA i;
  switch (type) {
    case TRI:
      return triangleIntersect(r, &i);
    case SPHERE:
      return sphereIntersect(r, &i);
    default:
      return false;
  }
}

__device__
bool PrimitiveGPU::intersect(Ray& r, IntersectionSoA* i)
{
  switch (type) {
    case TRI:
      return triangleIntersect(r, i);
    case SPHERE:
      return sphereIntersect(r, i);
    default:
      return false;
  }
}

__device__
BBox PrimitiveGPU::triangleGetBBox()
{
  Vector3D p1 = cuGlobals.points[inds.x];
  Vector3D p2 = cuGlobals.points[inds.y];
  Vector3D p3 = cuGlobals.points[inds.z];

  Vector3D min = Vector3D::make(fminf(p1.v.x, fminf(p2.v.x, p3.v.x)),
                                fminf(p1.v.y, fminf(p2.v.y, p3.v.y)),
                                fminf(p1.v.z, fminf(p2.v.z, p3.v.z)));

  Vector3D max = Vector3D::make(fmaxf(p1.v.x, fmaxf(p2.v.x, p3.v.x)),
                                fmaxf(p1.v.y, fmaxf(p2.v.y, p3.v.y)),
                                fmaxf(p1.v.z, fmaxf(p2.v.z, p3.v.z)));

  return BBox(min, max);
}

__device__
BBox PrimitiveGPU::sphereGetBBox()
{
  return BBox(o - Vector3D::make(r,r,r), o + Vector3D::make(r,r,r));
}

__device__
bool PrimitiveGPU::triangleIntersect(Ray& r, IntersectionSoA* i)
{
  Vector3D p1 = cuGlobals.points[inds.x];
  Vector3D p2 = cuGlobals.points[inds.y];
  Vector3D p3 = cuGlobals.points[inds.z];

  Vector3D e1= p2 - p1;
  Vector3D e2 = p3 - p1;
  Vector3D s = r.o - p1;
  Vector3D c1 = cross(e1, r.d), c2 = cross(e2, s);
  float denom = dot(c1, e2);
  float u = dot(c2, r.d) / denom, v = dot(c1, s) / denom;
  float t = dot(c2,e1) / denom;

  if (r.min_t <= t && t <= r.max_t && u >=0 && v >= 0 &&  (u + v) <= 1) {
    r.max_t = t;
    Vector3D n1 = cuGlobals.normals[inds.x];
    Vector3D n2 = cuGlobals.normals[inds.y];
    Vector3D n3 = cuGlobals.normals[inds.z];

    Vector3D n = u * n2 + v * n3 + (1 - u - v) * n1;
    if (dot(n, r.d) > 0) n *= -1;
    i->n[threadIdx.x] = n;
    i->t[threadIdx.x] = t;
    i->primitive[threadIdx.x] = this;
    i->bsdf[threadIdx.x] = bsdf;
    return true;
  }
  return false;

}

__device__
bool PrimitiveGPU::test(Ray& r, float& t1, float& t2) const {
  Vector3D l = r.o - o;

  float disc = dot(l, r.d)*dot(l, r.d) - dot(r.d, r.d) * (dot(l, l) - r2);
  if (disc < 0) return false;

  t1 = (-dot(l, r.d) - sqrtf(disc)) / dot(r.d, r.d);
  t2 = (-dot(l, r.d) + sqrtf(disc)) / dot(r.d, r.d);
  return true;
}


__device__
bool PrimitiveGPU::sphereIntersect(Ray& r, IntersectionSoA* i)
{
  float t1, t2;
  if (!test(r, t1, t2)) return false;

  if (r.min_t <= t1 && t1 <= r.max_t) {
    r.max_t = t1;
    i->n[threadIdx.x] = (r.o + r.d * t1 - o).unit();
    i->t[threadIdx.x] = t1;
    i->primitive[threadIdx.x] = this;
    i->bsdf[threadIdx.x] = bsdf;
    return true;
  }
  else if (r.min_t <= t2 && t2 <= r.max_t) {
    r.max_t = t2;
    i->n[threadIdx.x] = (r.o + r.d * t2 - o).unit();
    i->t[threadIdx.x] = t2;
    i->primitive[threadIdx.x] = this;
    i->bsdf[threadIdx.x] = bsdf;
    return true;
  }

  return false;
}

}
