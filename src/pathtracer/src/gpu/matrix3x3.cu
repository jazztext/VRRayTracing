#include "matrix3x3.h"

#include <iostream>
#include <cmath>

namespace VRRT {

__device__
float& Matrix3x3::operator()( int i, int j ) {
  return entries[j][i];
}

__device__
const float& Matrix3x3::operator()( int i, int j ) const {
  return entries[j][i];
}

__device__
Vector3D& Matrix3x3::operator[]( int j ) {
    return entries[j];
}

__device__
const Vector3D& Matrix3x3::operator[]( int j ) const {
  return entries[j];
}

__device__
void Matrix3x3::zero( float val ) {
  // sets all elements to val
  entries[0] = entries[1] = entries[2] = Vector3D::make( val, val, val );
}

__device__
float Matrix3x3::det( void ) const {
  const Matrix3x3& A( *this );

  return -A(0,2)*A(1,1)*A(2,0) + A(0,1)*A(1,2)*A(2,0) +
          A(0,2)*A(1,0)*A(2,1) - A(0,0)*A(1,2)*A(2,1) -
          A(0,1)*A(1,0)*A(2,2) + A(0,0)*A(1,1)*A(2,2) ;
}

__device__
float Matrix3x3::norm( void ) const {
  return sqrtf( entries[0].norm2() +
                entries[1].norm2() +
                entries[2].norm2() );
}

__device__
Matrix3x3 Matrix3x3::operator-( void ) const {

 // returns -A
  const Matrix3x3& A( *this );
  Matrix3x3 B;

  B(0,0) = -A(0,0); B(0,1) = -A(0,1); B(0,2) = -A(0,2);
  B(1,0) = -A(1,0); B(1,1) = -A(1,1); B(1,2) = -A(1,2);
  B(2,0) = -A(2,0); B(2,1) = -A(2,1); B(2,2) = -A(2,2);

  return B;
}

__device__
void Matrix3x3::operator+=( const Matrix3x3& B ) {

  Matrix3x3& A( *this );
  float* Aij = (float*) &A;
  const float* Bij = (const float*) &B;

  *Aij++ += *Bij++;
  *Aij++ += *Bij++;
  *Aij++ += *Bij++;
  *Aij++ += *Bij++;
  *Aij++ += *Bij++;
  *Aij++ += *Bij++;
  *Aij++ += *Bij++;
  *Aij++ += *Bij++;
  *Aij++ += *Bij++;
}

__device__
Matrix3x3 Matrix3x3::operator-( const Matrix3x3& B ) const {
  const Matrix3x3& A( *this );
  Matrix3x3 C;

  for( int i = 0; i < 3; i++ )
  for( int j = 0; j < 3; j++ )
  {
     C(i,j) = A(i,j) - B(i,j);
  }

  return C;
}

__device__
Matrix3x3 Matrix3x3::operator*( float c ) const {
  const Matrix3x3& A( *this );
  Matrix3x3 B;

  for( int i = 0; i < 3; i++ )
  for( int j = 0; j < 3; j++ )
  {
     B(i,j) = c*A(i,j);
  }

  return B;
}

__device__
Matrix3x3 operator*( float c, const Matrix3x3& A ) {

  Matrix3x3 cA;
  const float* Aij = (const float*) &A;
  float* cAij = (float*) &cA;

  *cAij++ = c * (*Aij++);
  *cAij++ = c * (*Aij++);
  *cAij++ = c * (*Aij++);
  *cAij++ = c * (*Aij++);
  *cAij++ = c * (*Aij++);
  *cAij++ = c * (*Aij++);
  *cAij++ = c * (*Aij++);
  *cAij++ = c * (*Aij++);
  *cAij++ = c * (*Aij++);

  return cA;
}

__device__
Matrix3x3 Matrix3x3::operator*( const Matrix3x3& B ) const {
  const Matrix3x3& A( *this );
  Matrix3x3 C;

  for( int i = 0; i < 3; i++ )
  for( int j = 0; j < 3; j++ )
  {
     C(i,j) = 0.;

     for( int k = 0; k < 3; k++ )
     {
        C(i,j) += A(i,k)*B(k,j);
     }
  }

  return C;
}

__device__
Vector3D Matrix3x3::operator*( const Vector3D& x ) const {
  return x[0]*entries[0] +
         x[1]*entries[1] +
         x[2]*entries[2] ;
}

__device__
Matrix3x3 Matrix3x3::T( void ) const {
  const Matrix3x3& A( *this );
  Matrix3x3 B;

  for( int i = 0; i < 3; i++ )
  for( int j = 0; j < 3; j++ )
  {
     B(i,j) = A(j,i);
  }

  return B;
}

__device__
Matrix3x3 Matrix3x3::inv( void ) const {
  const Matrix3x3& A( *this );
  Matrix3x3 B;

  B(0,0) = -A(1,2)*A(2,1) + A(1,1)*A(2,2); B(0,1) =  A(0,2)*A(2,1) - A(0,1)*A(2,2); B(0,2) = -A(0,2)*A(1,1) + A(0,1)*A(1,2);
  B(1,0) =  A(1,2)*A(2,0) - A(1,0)*A(2,2); B(1,1) = -A(0,2)*A(2,0) + A(0,0)*A(2,2); B(1,2) =  A(0,2)*A(1,0) - A(0,0)*A(1,2);
  B(2,0) = -A(1,1)*A(2,0) + A(1,0)*A(2,1); B(2,1) =  A(0,1)*A(2,0) - A(0,0)*A(2,1); B(2,2) = -A(0,1)*A(1,0) + A(0,0)*A(1,1);

  B /= det();

  return B;
}

__device__
void Matrix3x3::operator/=( float x ) {
  Matrix3x3& A( *this );
  float rx = 1./x;

  for( int i = 0; i < 3; i++ )
  for( int j = 0; j < 3; j++ )
  {
     A( i, j ) *= rx;
  }
}

__device__
Matrix3x3 Matrix3x3::identity( void ) {
  Matrix3x3 B;

  B(0,0) = 1.; B(0,1) = 0.; B(0,2) = 0.;
  B(1,0) = 0.; B(1,1) = 1.; B(1,2) = 0.;
  B(2,0) = 0.; B(2,1) = 0.; B(2,2) = 1.;

  return B;
}

__device__
Matrix3x3 Matrix3x3::crossProduct( const Vector3D& u ) {
  Matrix3x3 B;

  B(0,0) =   0.;  B(0,1) = -u.v.z;  B(0,2) =  u.v.y;
  B(1,0) =  u.v.z;  B(1,1) =   0.;  B(1,2) = -u.v.x;
  B(2,0) = -u.v.y;  B(2,1) =  u.v.x;  B(2,2) =   0.;

  return B;
}

__device__
Matrix3x3 outer( const Vector3D& u, const Vector3D& v ) {
  Matrix3x3 B;
  float* Bij = (float*) &B;

  *Bij++ = u.v.x*v.v.x;
  *Bij++ = u.v.y*v.v.x;
  *Bij++ = u.v.z*v.v.x;
  *Bij++ = u.v.x*v.v.y;
  *Bij++ = u.v.y*v.v.y;
  *Bij++ = u.v.z*v.v.y;
  *Bij++ = u.v.x*v.v.z;
  *Bij++ = u.v.y*v.v.z;
  *Bij++ = u.v.z*v.v.z;

  return B;
}

__device__
Vector3D& Matrix3x3::column( int i ) {
  return entries[i];
}

__device__
const Vector3D& Matrix3x3::column( int i ) const {
  return entries[i];
}

}
