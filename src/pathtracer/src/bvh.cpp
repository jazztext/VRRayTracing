#include "bvh.h"

#include "CMU462/CMU462.h"
#include "static_scene/triangle.h"

#include <iostream>
#include <stack>

using namespace std;

namespace CMU462 { namespace StaticScene {

double surfaceAreaCost(BVHNode *node, vector<BBox>& bboxes,
                       vector<int>& primCounts, int bin, int numBins)
{
  double sn = node->bb.surface_area();
  BBox bboxA, bboxB;
  int na = 0, nb = 0;
  for (int i = 0; i < numBins; i++) {
    if (i < bin) {
      bboxA.expand(bboxes[i]);
      na += primCounts[i];
    }
    else {
      bboxB.expand(bboxes[i]);
      nb += primCounts[i];
    }
  }
  if (na == 0 || nb == 0) return INF_D;
  return bboxA.surface_area() / sn * na + bboxB.surface_area() / sn * nb;
}

void BVHAccel::splitNode(BVHNode *node, int maxLeafSize)
{
  //base case
  if (node->range <= maxLeafSize) {
    return;
  }
  //choose plane to split along
  double smallestBins = INF_D;
  int numBins = 32;
  vector<int> binAssignments;
  int plane;
  double bestCost = INF_D;
  for (int n = 0; n < 3; n++) {
    double start = INF_D, end = -INF_D;
    for (int i = 0; i < node->range; i++) {
      double p = primitives[node->start + i]->get_bbox().centroid()[n];
      if (p < start) start = p;
      if (p > end) end = p;
    }
    double binSize = (end - start) / numBins * 1.0001;
    if (binSize == 0) {
    }
    else if (binSize > 0) {
    vector<BBox> binBBoxes(numBins);
    vector<int> binPrimCounts(numBins);
    vector<int> currentBinAssignments(node->range);
    for (unsigned i = 0; i < node->range; i++) {
      BBox bbox = primitives[node->start + i]->get_bbox();
      int bin = floor((bbox.centroid()[n] - start) / binSize);
      binBBoxes[bin].expand(bbox);
      binPrimCounts[bin]++;
      currentBinAssignments[i] = bin;
    }
    for (int i = 1; i < numBins; i++) {
      double cost = surfaceAreaCost(node, binBBoxes, binPrimCounts, i, numBins);
      if (cost < bestCost) {
        plane = i;
        bestCost = cost;
        binAssignments = currentBinAssignments;
      }
    }
    if (binSize < smallestBins) smallestBins = binSize;
    }
  }
  if (bestCost == INF_D) {
    for (int i = 0; i < node->range; i++) {
      if (i < node->range / 2) binAssignments.push_back(0);
      else binAssignments.push_back(1);
      plane = 1;
    }
  }
  //perform split
  //rearrange primitives
  int i = 0, j = node->range - 1;
  BBox lbox, rbox;
  while (i < j) {
    while (binAssignments[i] < plane && i <= j)
      lbox.expand(primitives[node->start + i++]->get_bbox());
    while (binAssignments[j] >= plane && j >= i)
      rbox.expand(primitives[node->start + j--]->get_bbox());
    if (i < j) {
      Primitive *tmp = primitives[node->start + i];
      primitives[node->start + i] = primitives[node->start + j];
      primitives[node->start + j] = tmp;
      int tmpBin = binAssignments[i];
      binAssignments[i] = binAssignments[j];
      binAssignments[j] = tmpBin;
    }
  }
  //create child nodes
  node->l = new BVHNode(lbox, node->start, i);
  node->r = new BVHNode(rbox, i + node->start, node->range - i);
  splitNode(node->l, maxLeafSize);
  splitNode(node->r, maxLeafSize);
}

BVHAccel::BVHAccel(const std::vector<Primitive *> &_primitives,
                   size_t max_leaf_size) {

  this->primitives = _primitives;

  // TODO:
  // Construct a BVH from the given vector of primitives and maximum leaf
  // size configuration. The starter code build a BVH aggregate with a
  // single leaf node (which is also the root) that encloses all the
  // primitives.

  BBox bb;
  for (size_t i = 0; i < primitives.size(); ++i) {
    bb.expand(primitives[i]->get_bbox());
  }

  root = new BVHNode(bb, 0, primitives.size());
  splitNode(root, max_leaf_size);
}

void deleteNode(BVHNode *node) {
  if (node->l != NULL) deleteNode(node->l);
  if (node->r != NULL) deleteNode(node->r);
  delete node;
}

BVHAccel::~BVHAccel() {
  // TODO:
  // Implement a proper destructor for your BVH accelerator aggregate
  deleteNode(root);
}

BBox BVHAccel::get_bbox() const {
  return root->bb;
}

bool BVHAccel::intersect(const Ray &ray) const {

  // TODO:
  // Implement ray - bvh aggregate intersection test. A ray intersects
  // with a BVH aggregate if and only if it intersects a primitive in
  // the BVH that is not an aggregate.

  Intersection i;

  return intersect(ray, &i);

}

bool BVHAccel::intersectNode(BVHNode *node, const Ray& ray, Intersection *i) const
{
  if (node->isLeaf()) {
    bool hit = false;
    for (int n = node->start; n < node->start + node->range; n++) {
      if (primitives[n]->intersect(ray, i)) hit = true;
    }
    return hit;
  }
  double minTL = ray.min_t, minTR = ray.min_t;
  double maxTL = ray.max_t, maxTR = ray.max_t;
  bool hitLeft = node->l->bb.intersect(ray, minTL, maxTL);
  bool hitRight = node->r->bb.intersect(ray, minTR, maxTR);
  BVHNode *first, *second;
  bool hitFirst, hitSecond;
  if (minTL < minTR) {
    first = node->l; second = node->r; hitFirst = hitLeft; hitSecond = hitRight;
  }
  else {
    first = node->r; second = node->l; hitFirst = hitRight; hitSecond = hitLeft;
  }
  bool hit = false;
  if (hitFirst && intersectNode(first, ray, i)) hit = true;
  if (hitSecond && minTR < ray.max_t && intersectNode(second, ray, i))
    hit = true;
  return hit;
}

bool BVHAccel::intersect(const Ray &ray, Intersection *i) const {

  // TODO:
  // Implement ray - bvh aggregate intersection test. A ray intersects
  // with a BVH aggregate if and only if it intersects a primitive in
  // the BVH that is not an aggregate. When an intersection does happen.
  // You should store the non-aggregate primitive in the intersection data
  // and not the BVH aggregate itself.

  bool returnVal = intersectNode(root, ray, i);
  return returnVal;

}

}  // namespace StaticScene
}  // namespace CMU462
