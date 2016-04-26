#ifndef __MATRIX4X4_H__
#define __MATRIX4X4_H__

#include "vector4D.h"

namespace VRRT {

/**
 * Defines a 4x4 matrix.
 * 4x4 matrices are also extremely useful in computer graphics.
 * Written by Bryce Summers on 9/10/2015.
 * Adapted from the Matrix3x3 class.
 *
 * EXTEND_ME : It might be nice to add some combined operations
 *             such as multiplying then adding,
 *             etc to increase arithmetic intensity.
 * I have taken the liberty of removing cross product functionality form 4D Matrices and Vectors.
 */
class Matrix4x4 {

  public:


  // The default constructor.
  __device__
  Matrix4x4(void) { }

  // Constructor for row major form data.
  // Transposes to the internal column major form.
  // REQUIRES: data should be of size 16.
  __device__
  Matrix4x4(double * data)
  {
    for( int i = 0; i < 4; i++ )
    for( int j = 0; j < 4; j++ )
    {
	  // Transpostion happens within the () query.
	  (*this)(i,j) = data[i*4 + j];
    }

  }


  /**
   * Sets all elements to val.
   */
  __device__
  void zero(double val = 0.0);

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
   * Returns a fresh 4x4 identity matrix.
   */
  __device__
  static Matrix4x4 identity( void );

  // No Cross products for 4 by 4 matrix.

  /**
   * Returns the ith column.
   */
  __device__
        Vector4D& column( int i );
  __device__
  const Vector4D& column( int i ) const;

  /**
   * Returns the transpose of A.
   */
  __device__
  Matrix4x4 T( void ) const;

  /**
   * Returns the inverse of A.
   */
  __device__
  Matrix4x4 inv( void ) const;

  // accesses element (i,j) of A using 0-based indexing
  // where (i, j) is (row, column).
  __device__
        double& operator()( int i, int j );
  __device__
  const double& operator()( int i, int j ) const;

  // accesses the ith column of A
  __device__
        Vector4D& operator[]( int i );
  __device__
  const Vector4D& operator[]( int i ) const;

  // increments by B
  __device__
  void operator+=( const Matrix4x4& B );

  // returns -A
  __device__
  Matrix4x4 operator-( void ) const;

  // returns A-B
  __device__
  Matrix4x4 operator-( const Matrix4x4& B ) const;

  // returns c*A
  __device__
  Matrix4x4 operator*( double c ) const;

  // returns A*B
  __device__
  Matrix4x4 operator*( const Matrix4x4& B ) const;

  // returns A*x
  __device__
  Vector4D operator*( const Vector4D& x ) const;

  // divides each element by x
  __device__
  void operator/=( double x );

  protected:

  // 4 by 4 matrices are represented by an array of 4 column vectors.
  Vector4D entries[4];

}; // class Matrix3x3

// returns the outer product of u and v.
__device__
Matrix4x4 outer( const Vector4D& u, const Vector4D& v );

// returns c*A
__device__
Matrix4x4 operator*( double c, const Matrix4x4& A );

} // namespace CMU462

#endif // CMU462_MATRIX4X4_H
