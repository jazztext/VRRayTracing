#ifndef __PRIMITIVE_GPU__
#define __PRIMITIVE_GPU__

#include "vector3D.h"
#include "bbox.h"
#include "intersection.h"

namespace VRRT {

class PrimitiveGPU {

 public:

  PrimitiveGPU() { }

  PrimitiveGPU(CMU462::Vector3D p1, CMU462::Vector3D p2, CMU462::Vector3D p3,
               CMU462::Vector3D n1, CMU462::Vector3D n2, CMU462::Vector3D n3,
               BSDF *bsdf);

  PrimitiveGPU(CMU462::Vector3D o, float r, BSDF *bsdf);

  /**
   * Get the world space bounding box of the primitive.
   * \return world space bounding box of the primitive
   */
  __device__
  BBox get_bbox();

  /**
   * Ray - Primitive intersection.
   * Check if the given ray intersects with the primitive, no intersection
   * information is stored.
   * \param r ray to test intersection with
   * \return true if the given ray intersects with the primitive,
             false otherwise
   */
  __device__
  bool intersect(Ray& r);

  /**
   * Ray - Primitive intersection 2.
   * Check if the given ray intersects with the primitive, if so, the input
   * intersection data is updated to contain intersection information for the
   * point of intersection.
   * \param r ray to test intersection with
   * \param i address to store intersection info
   * \return true if the given ray intersects with the primitive,
             false otherwise
   */
  __device__
  bool intersect(Ray& r, Intersection* i);

 private:
  __device__ BBox triangleGetBBox();
  __device__ BBox sphereGetBBox();
  __device__ bool triangleIntersect(Ray& r, Intersection* i);
  __device__ bool test(Ray& r, float& t1, float& t2) const;
  __device__ bool sphereIntersect(Ray& r, Intersection* i);

  enum {TRI, SPHERE};

  int type;
  Vector3D p1, p2, p3; //traingle points
  Vector3D n1, n2, n3; //triangle normals
  Vector3D o; ///< origin of the sphere
  float r;   ///< radius
  float r2;  ///< radius squared
  BSDF *bsdf;


};

}

#endif
