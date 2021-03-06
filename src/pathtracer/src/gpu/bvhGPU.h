#ifndef __BVH_GPU_H__
#define __BVH_GPU_H__

#include "misc.h"
#include "../bvh.h"

/**
 * A node in the BVH accelerator aggregate.
 * The accelerator uses a "flat tree" structure where all the primitives are
 * stored in one vector. A node in the data structure stores only the starting
 * index and the number of primitives in the node and uses this information to
 * index into the primitive vector for actual data. In this implementation all
 * primitives (index + range) are stored on leaf nodes. A leaf node has no child
 * node and its range should be no greater than the maximum leaf size used when
 * constructing the BVH.
 */

namespace VRRT {

struct BVHNodeGPU {

  BVHNodeGPU(CMU462::BBox bb, size_t start, size_t range)
      : start(start), range(range), l(NULL), r(NULL), minT(0)
  {
    this->bb.max.v.x = bb.max.x;
    this->bb.max.v.y = bb.max.y;
    this->bb.max.v.z = bb.max.z;
    this->bb.min.v.x = bb.min.x;
    this->bb.min.v.y = bb.min.y;
    this->bb.min.v.z = bb.min.z;
    this->bb.extent.v.x = bb.extent.x;
    this->bb.extent.v.y = bb.extent.y;
    this->bb.extent.v.z = bb.extent.z;
  }

  __device__
  inline bool isLeaf() const { return l == NULL && r == NULL; }

  BBox bb;        ///< bounding box of the node
  size_t start;   ///< start index into the primitive list
  size_t range;   ///< range of index into the primitive list
  BVHNodeGPU* l;     ///< left child node
  BVHNodeGPU* r;     ///< right child node
  float minT;
};

class BVHGPU {
 public:

   BVHGPU() { }

   BVHNodeGPU *flattenNode(CMU462::StaticScene::BVHNode *node,
                           std::vector<BVHNodeGPU> &flatNodes);

   BVHGPU(CMU462::StaticScene::BVHAccel *bvh, Vector3D **points,
          Vector3D **normals);

  ~BVHGPU();

  /**
   * Get the world space bounding box of the aggregate.
   * \return world space bounding box of the aggregate
   */
  __device__
  BBox get_bbox() const;

  /**
   * Ray - Aggregate intersection.
   * Check if the given ray intersects with the aggregate (any primitive in
   * the aggregate), no intersection information is stored.
   * \param r ray to test intersection with
   * \return true if the given ray intersects with the aggregate,
             false otherwise
   */
  __device__
  bool intersect(Ray& r) const;

  /**
   * Ray - Aggregate intersection 2.
   * Check if the given ray intersects with the aggregate (any primitive in
   * the aggregate). If so, the input intersection data is updated to contain
   * intersection information for the point of intersection. Note that the
   * intersected primitive entry in the intersection should be updated to
   * the actual primitive in the aggregate that the ray intersected with and
   * not the aggregate itself.
   * \param r ray to test intersection with
   * \param i address to store intersection info
   * \return true if the given ray intersects with the aggregate,
             false otherwise
   */
  __device__
  bool intersect(Ray& r, Intersection* i) const;

  PrimitiveGPU *primitives;
  int numPrims;
 private:
  BVHNodeGPU *nodes; ///< root node of the BVH
  int numNodes;
};

} // namespace StaticScene

#endif // CMU462_BVH_H
