#include "bsdf.h"
#include <iostream>

namespace VRRT {

__device__
void make_coord_space(Matrix3x3& o2w, const Vector3D& n) {

    Vector3D z = Vector3D(n.x, n.y, n.z);
    Vector3D h = z;
    if (fabsf(h.x) <= fabsf(h.y) && fabsf(h.x) <= fabsf(h.z)) h.x = 1.0;
    else if (fabsf(h.y) <= fabsf(h.x) && fabsf(h.y) <= fabsf(h.z)) h.y = 1.0;
    else h.z = 1.0;

    z.normalize();
    Vector3D y = cross(h, z);
    y.normalize();
    Vector3D x = cross(z, y);
    x.normalize();

    o2w[0] = x;
    o2w[1] = y;
    o2w[2] = z;

}

__device__
Spectrum BSDF::f(const Vector3D& wo, const Vector3D& wi)
{
  switch (t) {
    case DIFFUSE:
      return color * (1.f / PI);
    default:
      return Spectrum();
  }
}

__device__
Spectrum BSDF::sample_f (const Vector3D& wo, Vector3D* wi, float* pdf,
                         bool& inMat, curandState *state)
{
  switch (t) {
    case MIRROR:
      inMat = false;
      reflect(wo, wi);
      *pdf = 1;
      return color * (1 / wo.z);
    case GLASS:
      return glassSample(wo, wi, pdf, inMat, state);
    default:
      inMat = false;
      *wi = sampler.get_sample(state, pdf);
      return f(wo, *wi);
  }
}

__device__
Spectrum BSDF::get_emission()
{
  switch (t) {
    case EMISSION:
      return color * (1.0 / PI);
    default:
      return Spectrum();
  }
}

__device__
bool BSDF::is_delta()
{
  switch (t) {
    case MIRROR:
    case GLASS:
      return true;
    default:
      return false;
  }

}

__host__
BSDF *BSDF::copyToDev()
{
  BSDF *location;
  cudaCheckError( cudaMalloc(&location, sizeof(BSDF)) );
  cudaCheckError( cudaMemcpy(location, this, sizeof(BSDF),
                             cudaMemcpyHostToDevice) );
  return location;
}

__device__
Spectrum BSDF::glassSample(const Vector3D& wo, Vector3D* wi, float* pdf,
                           bool& inMat, curandState *state) {

  // Compute Fresnel coefficient and either reflect or refract based on it.
  float ni, nt;
  if (inMat) {
    ni = ior;
    nt = 1;
  }
  else {
    ni = 1;
    nt = ior;
  }
  Vector3D transmit;
  bool tir = !refract(wo, &transmit, ior, inMat);
  float cosThetaT = -wo.z, cosThetaI = transmit.z;
  float rPar = (nt * cosThetaI - ni * cosThetaT) / (nt * cosThetaI + ni * cosThetaT);
  float rPerp = (ni * cosThetaI - nt * cosThetaT) / (ni * cosThetaI + nt * cosThetaT);
  float fr = .5 * (powf(rPar, 2) + powf(rPerp, 2));
  if (tir || curand_uniform(state) < fr) { //reflect
    reflect(wo, wi);
    *pdf = 1;
    return color2 * (1 / fabs(wo.z));
  }
  else { //refract
    *wi = transmit;
    *pdf = 1;
    inMat = !inMat;
    return color * powf(ni / nt, 2) * (1 / (fabsf(cosThetaI)));
  }
}

__device__
void BSDF::reflect(const Vector3D& wo, Vector3D* wi) {

  // Implement reflection of wo about normal (0,0,1) and store result in wi.
  *wi = wo;
  wi->x *= -1;
  wi->y *= -1;

}

__device__
bool BSDF::refract(const Vector3D& wo, Vector3D* wi, float ior, bool inMat) {

  // Use Snell's Law to refract wo surface and store result ray in wi.
  // Return false if refraction does not occur due to total internal reflection
  // and true otherwise. When dot(wo,n) is positive, then wo corresponds to a
  // ray entering the surface through vacuum.

  float ni, nt;
  if (!inMat) {
    ni = 1;
    nt = ior;
  }
  else {
    ni = ior;
    nt = 1;
  }
  float radicand = 1 - powf(ni / nt, 2) * (1 - powf(wo.z, 2));
  if (radicand < 0) return false; //total internal reflection
  wi->z = (wo.z > 0) ? -sqrtf(radicand) : sqrtf(radicand);
  float scale = sqrtf(1 - radicand) / sqrtf(powf(wo.x, 2) + powf(wo.y, 2));
  wi->x = -scale * wo.x;
  wi->y = -scale * wo.y;
  return true;

}

} // namespace CMU462
