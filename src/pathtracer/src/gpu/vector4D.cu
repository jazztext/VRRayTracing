#include "vector4D.h"

namespace VRRT {

  __device__
  Vector3D Vector4D::to3D() {
    return Vector3D::make(x, y, z);
  }

  __device__
  Vector3D Vector4D::projectTo3D() {
    float invW = 1.0 / w;
    return Vector3D::make(x * invW, y * invW, z * invW);
  }

} // namespace CMU462
