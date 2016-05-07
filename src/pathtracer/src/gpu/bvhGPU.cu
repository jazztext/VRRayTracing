#include "bvhGPU.h"
#include "../static_scene/object.h"
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

BVHGPU::BVHGPU(CMU462::StaticScene::BVHAccel *bvh, Vector3D **points,
               Vector3D **normals)
{
  std::vector<PrimitiveGPU> cpuPrims;
  std::unordered_map<BSDF *, BSDF *> bsdfs;
  std::unordered_map<const CMU462::StaticScene::Mesh *, int> meshes;

  //copy bsdfs and primitives to GPU
  int nVerts = 0;
  for (int i = 0; i < bvh->primitives.size(); i++) {
    cpuPrims.push_back(bvh->primitives[i]->toGPU(bsdfs, meshes, nVerts));
  }
  cudaCheckError( cudaDeviceSynchronize() );
  cudaCheckError( cudaMalloc(&primitives, sizeof(PrimitiveGPU) * cpuPrims.size()) );
  cudaCheckError( cudaMemcpy(primitives, cpuPrims.data(), sizeof(PrimitiveGPU)*cpuPrims.size(),
             cudaMemcpyHostToDevice) );
  numPrims = cpuPrims.size();

  //copy points and normals to GPU
  cudaCheckError( cudaMalloc(points, sizeof(Vector3D) * nVerts) );
  cudaCheckError( cudaMalloc(normals, sizeof(Vector3D) * nVerts) );
  for (std::pair<const CMU462::StaticScene::Mesh *, int> m : meshes) {
    std::vector<Vector3D> newPoints, newNormals;
    for (int i = 0; i < m.first->nVerts; i++) {
      newPoints.push_back(Vector3D::make(m.first->positions[i]));
      newNormals.push_back(Vector3D::make(m.first->normals[i]));
    }
    cudaCheckError( cudaMemcpy(*points + m.second, newPoints.data(), sizeof(Vector3D) * m.first->nVerts, cudaMemcpyHostToDevice) );
    cudaCheckError( cudaMemcpy(*normals + m.second, newNormals.data(), sizeof(Vector3D) * m.first->nVerts, cudaMemcpyHostToDevice) );

  }

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
  __shared__ IntersectionSoA tmpI;
  BVHNodeGPU *node = nodes;
  BVHNodeGPU *stack[50];
  int top = 0;
  node->minT = 0;
  pushStack(node, stack, top);
  bool hit = false;
  while (top > 0) {
    node = popStack(stack, top);
    if (node->minT >= ray.max_t) continue;
    while (!node->isLeaf()) {
      float minTL = ray.min_t, minTR = ray.min_t;
      float maxTL = ray.max_t, maxTR = ray.max_t;
      bool hitLeft = node->l->bb.intersect(ray, minTL, maxTL);
      bool hitRight = node->r->bb.intersect(ray, minTR, maxTR);
      if (hitLeft && hitRight) {
        if (minTL < minTR) {
          node = node->l;
          node->r->minT = minTR;
          pushStack(node->r, stack, top);
        }
        else {
          node = node->r;
          node->l->minT = minTL;
          pushStack(node->l, stack, top);
        }
      }
      else if (hitLeft) {
        node = node->l;
      }
      else if (hitRight) {
        node = node->r;
      }
      else if (top == 0) break;
      else {
        while (top > 0) {
          node = popStack(stack, top);
          if (node->minT < ray.max_t) break;
        }
        if (node->minT >= ray.max_t) break;
      }
    }
    if (!node->isLeaf()) break;
    for (int n = node->start; n < node->start + node->range; n++) {
      if (primitives[n].intersect(ray, &tmpI)) {
        hit = true;
      }
    }
  }
  i->n = tmpI.n[threadIdx.x];
  i->t = tmpI.t[threadIdx.x];
  i->primitive = tmpI.primitive[threadIdx.x];
  i->bsdf = tmpI.bsdf[threadIdx.x];
  return hit;
}

}
