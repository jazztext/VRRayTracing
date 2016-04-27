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
  float x, y, z;

  /**
   * Constructor.
   * Initializes tp vector (0,0,0).
   */
  __device__
  Vector3D() : x( 0.f ), y( 0.f ), z( 0.f ) { }

  /**
   * Constructor.
   * Initializes to vector (x,y,z).
   */
  __device__
  Vector3D( float x, float y, float z) : x( x ), y( y ), z( z ) { }

  /**
   * Constructor.
   * Initializes to vector (c,c,c)
   */
  __device__
  Vector3D( float c ) : x( c ), y( c ), z( c ) { }

  /**
   * Constructor.
   * Initializes from existing vector
   */
  __device__
  Vector3D( const Vector3D& v ) : x( v.x ), y( v.y ), z( v.z ) { }

  __host__ __device__
  Vector3D( const CMU462::Vector3D& v) : x(v.x), y(v.y), z(v.z) { }

  // returns reference to the specified component (0-based indexing: x, y, z)
  __device__
  inline float& operator[] ( const int& index ) {
    return ( &x )[ index ];
  }

  // returns const reference to the specified component (0-based indexing: x, y, z)
  __device__
  inline const float& operator[] ( const int& index ) const {
    return ( &x )[ index ];
  }

  __device__
  inline bool operator==( const Vector3D& v) const {
    return v.x == x && v.y == y && v.z == z;
  }

  // negation
  __device__
  inline Vector3D operator-( void ) const {
    return Vector3D( -x, -y, -z );
  }

  // addition
  __device__
  inline Vector3D operator+( const Vector3D& v ) const {
    return Vector3D( x + v.x, y + v.y, z + v.z );
  }

  // subtraction
  __device__
  inline Vector3D operator-( const Vector3D& v ) const {
    return Vector3D( x - v.x, y - v.y, z - v.z );
  }

  // right scalar multiplication
  __device__
  inline Vector3D operator*( const float& c ) const {
    return Vector3D( x * c, y * c, z * c );
  }

  // scalar division
  __device__
  inline Vector3D operator/( const float& c ) const {
    const float rc = 1.0/c;
    return Vector3D( rc * x, rc * y, rc * z );
  }

  // addition / assignment
  __device__
  inline void operator+=( const Vector3D& v ) {
    x += v.x; y += v.y; z += v.z;
  }

  // subtraction / assignment
  __device__
  inline void operator-=( const Vector3D& v ) {
    x -= v.x; y -= v.y; z -= v.z;
  }

  // scalar multiplication / assignment
  __device__
  inline void operator*=( const float& c ) {
    x *= c; y *= c; z *= c;
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
    return sqrtf( x*x + y*y + z*z );
  }

  /**
   * Returns Euclidean length squared.
   */
  __device__
  inline float norm2( void ) const {
    return x*x + y*y + z*z;
  }

  /**
   * Returns unit vector.
   */
  __device__
  inline Vector3D unit( void ) const {
    float rNorm = 1. / sqrtf( x*x + y*y + z*z );
    return Vector3D( rNorm*x, rNorm*y, rNorm*z );
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
  return Vector3D( c * v.x, c * v.y, c * v.z );
}

// dot product (a.k.a. inner or scalar product)
__device__
inline float dot( const Vector3D& u, const Vector3D& v ) {
  return u.x*v.x + u.y*v.y + u.z*v.z ;
}

// cross product
__device__
inline Vector3D cross( const Vector3D& u, const Vector3D& v ) {
  return Vector3D( u.y*v.z - u.z*v.y,
                   u.z*v.x - u.x*v.z,
                   u.x*v.y - u.y*v.x );
}

} // namespace CMU462

#endif // CMU462_VECTOR3D_H
