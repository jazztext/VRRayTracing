#ifndef __MATRIX3X3_H__
#define __MATRIX3X3_H__

#include "misc.h"
#include "vector3D.h"
#include "CMU462/matrix3x3.h"

namespace VRRT {

/**
 * Defines a 3x3 matrix.
 * 3x3 matrices are extremely useful in computer graphics.
 */
class Matrix3x3 {

  public:

  // The default constructor.
  __device__
  Matrix3x3(void) { }

  __host__ __device__
  Matrix3x3(const CMU462::Matrix3x3& m)
  {
    for (int i = 0; i < 3; i++) {
      entries[i] = Vector3D(m.entries[i]);
    }
  }

  // Constructor for row major form data.
  // Transposes to the internal column major form.
  // REQUIRES: data should be of size 9 for a 3 by 3 matrix..
  __device__
  Matrix3x3(double * data)
  {
    for( int i = 0; i < 3; i++ ) {
      for( int j = 0; j < 3; j++ ) {
	        // Transpostion happens within the () query.
	        (*this)(i,j) = data[i*3 + j];
      }
    }
  }

  /**
   * Sets all elements to val.
   */
  __device__
  void zero(double val = 0.0 );

  /**
   * Returns the determinant of A.
   */
  __device__
  double det( void ) const;

  /**
   * Returns the Frobenius norm of A.
   */
  __device__
  double norm( void ) const;

  /**
   * Returns the 3x3 identity matrix.
   */
  __device__
  static Matrix3x3 identity( void );

  /**
   * Returns a matrix representing the (left) cross product with u.
   */
  __device__
  static Matrix3x3 crossProduct( const Vector3D& u );

  /**
   * Returns the ith column.
   */
  __device__
        Vector3D& column( int i );
  __device__
  const Vector3D& column( int i ) const;

  /**
   * Returns the transpose of A.
   */
  __device__
  Matrix3x3 T( void ) const;

  /**
   * Returns the inverse of A.
   */
  __device__
  Matrix3x3 inv( void ) const;

  // accesses element (i,j) of A using 0-based indexing
  __device__
        double& operator()( int i, int j );
  __device__
  const double& operator()( int i, int j ) const;

  // accesses the ith column of A
  __device__
        Vector3D& operator[]( int i );
  __device__
  const Vector3D& operator[]( int i ) const;

  // increments by B
  __device__
  void operator+=( const Matrix3x3& B );

  // returns -A
  __device__
  Matrix3x3 operator-( void ) const;

  // returns A-B
  __device__
  Matrix3x3 operator-( const Matrix3x3& B ) const;

  // returns c*A
  __device__
  Matrix3x3 operator*( double c ) const;

  // returns A*B
  __device__
  Matrix3x3 operator*( const Matrix3x3& B ) const;

  // returns A*x
  __device__
  Vector3D operator*( const Vector3D& x ) const;

  // divides each element by x
  __device__
  void operator/=( double x );

  protected:

  // column vectors
  Vector3D entries[3];

}; // class Matrix3x3

// returns the outer product of u and v
__device__
Matrix3x3 outer( const Vector3D& u, const Vector3D& v );

// returns c*A
__device__
Matrix3x3 operator*( double c, const Matrix3x3& A );

} // namespace CMU462

#endif // CMU462_MATRIX3X3_H
