#ifndef __BBOX_H__
#define __BBOX_H__

#include "vector3D.h"
#include "ray.h"

namespace VRRT {

/**
 * Axis-aligned bounding box.
 * An AABB is given by two positions in space, the min and the max. An addition
 * component, the extent of the bounding box is stored as it is useful in a lot
 * of the operations on bounding boxes.
 */
struct BBox {

  Vector3D max;	    ///< min corner of the bounding box
  Vector3D min;	    ///< max corner of the bounding box
  Vector3D extent;  ///< extent of the bounding box (min -> max)

  /**
   * Constructor.
   * The default constructor creates a new bounding box which contains no
   * points.
   */
  __device__
  BBox() {
    max = Vector3D(-infy(), -infy(), -infy());
    min = Vector3D( infy(),  infy(),  infy());
    extent = max - min;
  }

  /**
   * Constructor.
   * Creates a bounding box that includes a single point.
   */
  __device__
  BBox(const Vector3D& p) : min(p), max(p) { extent = max - min; }

  /**
   * Constructor.
   * Creates a bounding box with given bounds.
   * \param min the min corner
   * \param max the max corner
   */
  __device__
  BBox(const Vector3D& min, const Vector3D& max) :
       min(min), max(max), extent(max - min) { }

  __host__
  BBox(const CMU462::Vector3D& min, const CMU462::Vector3D& max) :
       min(min), max(max), extent(max - min) { }

  /**
   * Constructor.
   * Creates a bounding box with given bounds (component wise).
   */
  __device__
  BBox(const float minX, const float minY, const float minZ,
       const float maxX, const float maxY, const float maxZ) {
    min = Vector3D(minX, minY, minZ);
    max = Vector3D(maxX, maxY, maxZ);
		extent = max - min;
  }

  /**
   * Expand the bounding box to include another (union).
   * If the given bounding box is contained within *this*, nothing happens.
   * Otherwise *this* is expanded to the minimum volume that contains the
   * given input.
   * \param bbox the bounding box to be included
   */
  __device__
  void expand(const BBox& bbox) {
    min.x = fminf(min.x, bbox.min.x);
    min.y = fminf(min.y, bbox.min.y);
    min.z = fminf(min.z, bbox.min.z);
    max.x = fmaxf(max.x, bbox.max.x);
    max.y = fmaxf(max.y, bbox.max.y);
    max.z = fmaxf(max.z, bbox.max.z);
    extent = max - min;
  }

  /**
   * Expand the bounding box to include a new point in space.
   * If the given point is already inside *this*, nothing happens.
   * Otherwise *this* is expanded to a minimum volume that contains the given
   * point.
   * \param p the point to be included
   */
  __device__
  void expand(const Vector3D& p) {
    min.x = fminf(min.x, p.x);
    min.y = fminf(min.y, p.y);
    min.z = fminf(min.z, p.z);
    max.x = fmaxf(max.x, p.x);
    max.y = fmaxf(max.y, p.y);
    max.z = fmaxf(max.z, p.z);
    extent = max - min;
  }

  __device__
  Vector3D centroid() const {
    return (min + max) / 2;
  }

  /**
   * Compute the surface area of the bounding box.
   * \return surface area of the bounding box.
   */
  __device__
  float surface_area() const {
    if (empty()) return 0.0;
    return 2 * (extent.x * extent.z +
                extent.x * extent.y +
                extent.y * extent.z);
  }

  /**
   * Check if bounding box is empty.
   * Bounding box that has no size is considered empty. Note that since
   * bounding box are used for objects with positive volumes, a bounding
   * box of zero size (empty, or contains a single vertex) are considered
   * empty.
   */
  __device__
  bool empty() const {
    return min.x > max.x || min.y > max.y || min.z > max.z;
  }

  /**
   * Ray - bbox intersection.
   * Intersects ray with bounding box, does not store shading information.
   * \param r the ray to intersect with
   * \param t0 lower bound of intersection time
   * \param t1 upper bound of intersection time
   */
  __device__
  bool intersect(Ray& r, float& t0, float& t1) const;

};

} // namespace CMU462

#endif // CMU462_BBOX_H
