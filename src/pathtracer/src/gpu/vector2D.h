#ifndef __VECTOR2D_H__
#define __VECTOR2D_H__

namespace VRRT {

/**
 * Defines 2D vectors.
 */
class Vector2D {
 public:

  // components
  float x, y;

  /**
   * Constructor.
   * Initializes to vector (0,0).
   */
  __device__
  Vector2D() : x( 0.0 ), y( 0.0 ) { }

  /**
   * Constructor.
   * Initializes to vector (a,b).
   */
  __device__
  Vector2D( float x, float y ) : x( x ), y( y ) { }

  /**
   * Constructor.
   * Copy constructor. Creates a copy of the given vector.
   */
  __device__
  Vector2D( const Vector2D& v ) : x( v.x ), y( v.y ) { }

  // additive inverse
  __device__
  inline Vector2D operator-( void ) const {
    return Vector2D( -x, -y );
  }

  // addition
  __device__
  inline Vector2D operator+( const Vector2D& v ) const {
    Vector2D u = *this;
    u += v;
    return u;
  }

  // subtraction
  __device__
  inline Vector2D operator-( const Vector2D& v ) const {
    Vector2D u = *this;
    u -= v;
    return u;
  }

  // right scalar multiplication
  __device__
  inline Vector2D operator*( float r ) const {
    Vector2D vr = *this;
    vr *= r;
    return vr;
  }

  // scalar division
  __device__
  inline Vector2D operator/( float r ) const {
    Vector2D vr = *this;
    vr /= r;
    return vr;
  }

  // add v
  __device__
  inline void operator+=( const Vector2D& v ) {
    x += v.x;
    y += v.y;
  }

  // subtract v
  __device__
  inline void operator-=( const Vector2D& v ) {
    x -= v.x;
    y -= v.y;
  }

  // scalar multiply by r
  __device__
  inline void operator*=( float r ) {
    x *= r;
    y *= r;
  }

  // scalar divide by r
  __device__
  inline void operator/=( float r ) {
    x /= r;
    y /= r;
  }

  /**
   * Returns norm.
   */
  __device__
  inline float norm( void ) const {
    return sqrtf( x*x + y*y );
  }

  /**
   * Returns norm squared.
   */
  __device__
  inline float norm2( void ) const {
    return x*x + y*y;
  }

  /**
   * Returns unit vector parallel to this one.
   */
  __device__
  inline Vector2D unit( void ) const {
    return *this / this->norm();
  }


}; // clasd Vector2D

// left scalar multiplication
__device__
inline Vector2D operator*( float r, const Vector2D& v ) {
   return v*r;
}

// inner product
__device__
inline float dot( const Vector2D& v1, const Vector2D& v2 ) {
  return v1.x*v2.x + v1.y*v2.y;
}

// cross product
__device__
inline float cross( const Vector2D& v1, const Vector2D& v2 ) {
  return v1.x*v2.y - v1.y*v2.x;
}

} // namespace CMU462

#endif // CMU462_VECTOR2D_H
