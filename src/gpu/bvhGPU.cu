#include "bvhGPU.h"
#include <iostream>
#include <unordered_map>
#include <thrust/device_vector.h>

namespace VRRT {

BVHNodeGPU *BVHGPU::flattenNode(CMU462::StaticScene::BVHNode *node,
                                std::vector<BVHNodeGPU> &flatNodes)
{
  flatNodes.emplace_back(node->bb, node->start, node->range);
  BVHNodeGPU *currNode = &flatNodes.back();
  if (!node->isLeaf()) {
    currNode->l = flattenNode(node->l, flatNodes);
    currNode->r = flattenNode(node->r, flatNodes);
  }
  return currNode;
}

BVHGPU::BVHGPU(CMU462::StaticScene::BVHAccel *bvh)
{
  std::vector<PrimitiveGPU> cpuPrims;
  std::unordered_map<BSDF *, BSDF *> bsdfs;

  //copy bsdfs and primitives to GPU
  for (int i = 0; i < bvh->primitives.size(); i++) {
    cpuPrims.push_back(bvh->primitives[i]->toGPU(bsdfs));
  }
  cudaCheckError( cudaDeviceSynchronize() );
  cudaCheckError( cudaMalloc(&primitives, sizeof(PrimitiveGPU) * cpuPrims.size()) );
  cudaCheckError( cudaMemcpy(primitives, cpuPrims.data(), sizeof(PrimitiveGPU)*cpuPrims.size(),
             cudaMemcpyHostToDevice) );

  //flatten node structure and copy to GPU
  std::vector<BVHNodeGPU> flatNodes;
  flattenNode(bvh->root, flatNodes);
  cudaCheckError( cudaMalloc(&nodes, sizeof(BVHNodeGPU) * flatNodes.size()) );
  cudaCheckError( cudaMemcpy(nodes, flatNodes.data(), sizeof(BVHNodeGPU) * flatNodes.size(),
             cudaMemcpyHostToDevice) );
  numNodes = flatNodes.size();
}

BVHGPU::~BVHGPU() {
  //cudaCheckError( cudaFree(nodes) );
  //cudaCheckError( cudaFree(primitives) );
}

__device__
BBox BVHGPU::get_bbox() const {
  return nodes[0].bb;
}

__device__
bool BVHGPU::intersect(Ray &ray) const {

  // TODO:
  // Implement ray - bvh aggregate intersection test. A ray intersects
  // with a BVH aggregate if and only if it intersects a primitive in
  // the BVH that is not an aggregate.

  Intersection i;

  return intersect(ray, &i);

}

__device__ inline void pushStack(BVHNodeGPU *node, BVHNodeGPU **stack, int &top)
{
  stack[top] = node;
  top++;
}

__device__ inline BVHNodeGPU *popStack(BVHNodeGPU **stack, int &top)
{
  return stack[--top];
}

__device__
bool BVHGPU::intersect(Ray &ray, Intersection *i) const
{
  BVHNodeGPU *node = nodes;
  BVHNodeGPU *stack[50];
  int top = 0;
  node->minT = 0;
  pushStack(node, stack, top);
  bool hit = false;
  while (top > 0) {
    node = popStack(stack, top);
    if (node->minT >= ray.max_t) continue;
    if (node->isLeaf()) {
      for (int n = node->start; n < node->start + node->range; n++) {
        if (primitives[n].intersect(ray, i)) hit = true;
      }
      continue;
    }
    float minTL = ray.min_t, minTR = ray.min_t;
    float maxTL = ray.max_t, maxTR = ray.max_t;
    bool hitLeft = node->l->bb.intersect(ray, minTL, maxTL);
    bool hitRight = node->r->bb.intersect(ray, minTR, maxTR);
    BVHNodeGPU *first, *second;
    bool hitFirst, hitSecond;
    if (minTL < minTR) {
      first = node->l; second = node->r; hitFirst = hitLeft; hitSecond = hitRight;
    }
    else {
      first = node->r; second = node->l; hitFirst = hitRight; hitSecond = hitLeft;
    }
    if (hitFirst) {
      first->minT = minTL;
      pushStack(first, stack, top);
    }
    if (hitSecond) {
      second->minT = minTR;
      pushStack(second, stack, top);
    }
  }
  return hit;
}

}
