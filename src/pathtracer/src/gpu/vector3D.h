#ifndef __VECTOR3D_H__
#define __VECTOR3D_H__

#include "misc.h"
#include "CMU462/vector3D.h"
#include <stdio.h>

namespace VRRT {

/**
 * Defines 3D vectors.
 */
class Vector3D {
 public:

  // components
  float3 v;

  /**
   * Constructor.
   * Initializes tp vector (0,0,0).
   */
  __device__
  static Vector3D make()
  {
    Vector3D v;
    v.v = make_float3(0.f, 0.f, 0.f);
    return v;
  }

  /**
   * Constructor.
   * Initializes to vector (x,y,z).
   */
  __device__
  static Vector3D make(float x, float y, float z)
  {
    Vector3D v;
    v.v = make_float3(x, y, z);
    return v;
  }

  /**
   * Constructor.
   * Initializes to vector (c,c,c)
   */
  __device__
  static Vector3D make(float c)
  {
    Vector3D v;
    v.v = make_float3(c, c, c);
    return v;
  }


  /**
   * Constructor.
   * Initializes from existing vector
   */
  __device__
  static Vector3D make(const Vector3D& other)
  {
    Vector3D v;
    v.v = other.v;
    return v;
  }

  __host__ __device__
  static Vector3D make(const CMU462::Vector3D& other)
  {
    Vector3D v;
    v.v = make_float3(other.x, other.y, other.z) ;
    return v;
  }

  // returns reference to the specified component (0-based indexing: x, y, z)
  __device__
  inline float& operator[] ( const int& index ) {
    return ((float *) &v)[index];
  }

  // returns const reference to the specified component (0-based indexing: x, y, z)
  __device__
  inline const float& operator[] ( const int& index ) const {
    return ((float *) &v)[index];
  }

  __device__
  inline bool operator==( const Vector3D& other) const {
    return (other.v.x == v.x && other.v.y == v.y && other.v.z == v.z);
  }

  // negation
  __device__
  inline Vector3D operator-( void ) const {
    return make(-v.x, -v.y, -v.z);
  }

  // addition
  __device__
  inline Vector3D operator+( const Vector3D& other ) const {
    return make(other.v.x + v.x, other.v.y + v.y, other.v.z + v.z );
  }

  // subtraction
  __device__
  inline Vector3D operator-( const Vector3D& other ) const {
    return make(v.x - other.v.x, v.y - other.v.y, v.z - other.v.z );
  }

  // right scalar multiplication
  __device__
  inline Vector3D operator*( const float& c ) const {
    return make( v.x * c, v.y * c, v.z * c );
  }

  // scalar division
  __device__
  inline Vector3D operator/( const float& c ) const {
    const float rc = 1.0/c;
    return make( rc * v.x, rc * v.y, rc * v.z );
  }

  // addition / assignment
  __device__
  inline void operator+=( const Vector3D& other ) {
    v.x += other.v.x; v.y += other.v.y; v.z += other.v.z;
  }

  // subtraction / assignment
  __device__
  inline void operator-=( const Vector3D& other ) {
    v.x -= other.v.x; v.y -= other.v.y; v.z -= other.v.z;
  }

  // scalar multiplication / assignment
  __device__
  inline void operator*=( const float& c ) {
    v.x *= c; v.y *= c; v.z *= c;
  }

  // scalar division / assignment
  __device__
  inline void operator/=( const float& c ) {
    (*this) *= ( 1./c );
  }

  /**
   * Returns Euclidean length.
   */
  __device__
  inline float norm( void ) const {
    return sqrtf( v.x*v.x + v.y*v.y + v.z*v.z );
  }

  /**
   * Returns Euclidean length squared.
   */
  __device__
  inline float norm2( void ) const {
    return v.x*v.x + v.y*v.y + v.z*v.z;
  }

  /**
   * Returns unit vector.
   */
  __device__
  inline Vector3D unit( void ) const {
    float rNorm = 1. / sqrtf( v.x*v.x + v.y*v.y + v.z*v.z );
    return make( rNorm*v.x, rNorm*v.y, rNorm*v.z );
  }

  /**
   * Divides by Euclidean length.
   */
  __device__
  inline void normalize( void ) {
    (*this) /= norm();
  }

}; // class Vector3D

// left scalar multiplication
__device__
inline Vector3D operator* ( const float& c, const Vector3D& v ) {
  return Vector3D::make( c * v.v.x, c * v.v.y, c * v.v.z );
}

// dot product (a.k.a. inner or scalar product)
__device__
inline float dot( const Vector3D& u, const Vector3D& v ) {
  return u.v.x*v.v.x + u.v.y*v.v.y + u.v.z*v.v.z ;
}

// cross product
__device__
inline Vector3D cross( const Vector3D& u, const Vector3D& v ) {
  return Vector3D::make( u.v.y*v.v.z - u.v.z*v.v.y,
                         u.v.z*v.v.x - u.v.x*v.v.z,
                         u.v.x*v.v.y - u.v.y*v.v.x );
}

} // namespace CMU462

#endif // CMU462_VECTOR3D_H
