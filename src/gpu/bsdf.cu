#include "bsdf.h"
#include <iostream>

namespace VRRT {

__device__
void make_coord_space(Matrix3x3& o2w, const Vector3D& n) {

    Vector3D z = Vector3D(n.x, n.y, n.z);
    Vector3D h = z;
    if (fabs(h.x) <= fabs(h.y) && fabs(h.x) <= fabs(h.z)) h.x = 1.0;
    else if (fabs(h.y) <= fabs(h.x) && fabs(h.y) <= fabs(h.z)) h.y = 1.0;
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

// Diffuse BSDF //

__device__
Spectrum DiffuseBSDF::f(const Vector3D& wo, const Vector3D& wi) {
  return albedo * (1.0 / PI);
}

__device__
Spectrum DiffuseBSDF::sample_f(const Vector3D& wo, Vector3D* wi, float* pdf,
                               bool& inMat, curandState *state) {
  inMat = false;
  *wi = sampler.get_sample(state, pdf);
  return f(wo, *wi);
}

__global__ void copyDiffuseBSDF(Spectrum a, BSDF **dest)
{
  *dest = new DiffuseBSDF(a);
}

__host__
BSDF *DiffuseBSDF::copyToDev()
{
  BSDF **location;
  cudaCheckError( cudaMalloc(&location, sizeof(BSDF *)) );
  copyDiffuseBSDF<<<1, 1>>>(albedo, location);
  cudaCheckError( cudaGetLastError() );
  cudaCheckError( cudaDeviceSynchronize() );
  BSDF *devLoc;
  cudaMemcpy(&devLoc, location, sizeof(BSDF *), cudaMemcpyDeviceToHost);
  return devLoc;
}


// Mirror BSDF //

__device__
Spectrum MirrorBSDF::f(const Vector3D& wo, const Vector3D& wi) {
  return Spectrum(0, 0, 0);
}

__device__
Spectrum MirrorBSDF::sample_f(const Vector3D& wo, Vector3D* wi, float* pdf,
                              bool& inMat, curandState *state) {

  // TODO:
  // Implement MirrorBSDF
  inMat = false;
  reflect(wo, wi);
  *pdf = 1;
  return reflectance * (1 / wo.z);
}

__global__ void copyMirrorBSDF(MirrorBSDF bsdf, BSDF **dest)
{
  *dest = new MirrorBSDF(bsdf);
}

__host__
BSDF *MirrorBSDF::copyToDev()
{
  BSDF **location;
  cudaCheckError( cudaMalloc(&location, sizeof(BSDF *)) );
  copyMirrorBSDF<<<1, 1>>>(*this, location);
  cudaCheckError( cudaGetLastError() );
  cudaCheckError( cudaDeviceSynchronize() );
  BSDF *devLoc;
  cudaMemcpy(&devLoc, location, sizeof(BSDF *), cudaMemcpyDeviceToHost);
  return devLoc;
}


// Glossy BSDF //

/*
Spectrum GlossyBSDF::f(const Vector3D& wo, const Vector3D& wi) {
  return Spectrum();
}

Spectrum GlossyBSDF::sample_f(const Vector3D& wo, Vector3D* wi, float* pdf) {
  *pdf = 1.0f;
  return reflect(wo, wi, reflectance);
}
*/

// Refraction BSDF //

__device__
Spectrum RefractionBSDF::f(const Vector3D& wo, const Vector3D& wi) {
  return Spectrum();
}

__device__
Spectrum RefractionBSDF::sample_f(const Vector3D& wo, Vector3D* wi, float* pdf,
                                  bool& inMat, curandState *state) {

  // TODO:
  // Implement RefractionBSDF

  return Spectrum();
}

__global__ void copyRefractionBSDF(RefractionBSDF bsdf, BSDF **dest)
{
  *dest = new RefractionBSDF(bsdf);
}

__host__
BSDF *RefractionBSDF::copyToDev()
{
  BSDF **location;
  cudaCheckError( cudaMalloc(&location, sizeof(BSDF *)) );
  copyRefractionBSDF<<<1, 1>>>(*this, location);
  cudaCheckError( cudaGetLastError() );
  cudaCheckError( cudaDeviceSynchronize() );
  BSDF *devLoc;
  cudaMemcpy(&devLoc, location, sizeof(BSDF *), cudaMemcpyDeviceToHost);
  return devLoc;
}


// Glass BSDF //

__device__
Spectrum GlassBSDF::f(const Vector3D& wo, const Vector3D& wi) {
  return Spectrum();
}

__device__
Spectrum GlassBSDF::sample_f(const Vector3D& wo, Vector3D* wi, float* pdf,
                             bool& inMat, curandState *state) {

  // TODO:
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
    return reflectance * (1 / fabs(wo.z));
  }
  else { //refract
    *wi = transmit;
    *pdf = 1;
    inMat = !inMat;
    return transmittance * powf(ni / nt, 2) * (1 / (fabs(cosThetaI)));
  }
}


__device__
void BSDF::reflect(const Vector3D& wo, Vector3D* wi) {

  // TODO:
  // Implement reflection of wo about normal (0,0,1) and store result in wi.
  *wi = wo;
  wi->x *= -1;
  wi->y *= -1;

}

__device__
bool BSDF::refract(const Vector3D& wo, Vector3D* wi, float ior, bool inMat) {

  // TODO:
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
  wi->z = (wo.z > 0) ? -sqrt(radicand) : sqrt(radicand);
  float scale = sqrt(1 - radicand) / sqrt(powf(wo.x, 2) + powf(wo.y, 2));
  wi->x = -scale * wo.x;
  wi->y = -scale * wo.y;
  return true;

}

__global__ void copyGlassBSDF(GlassBSDF bsdf, BSDF **dest)
{
  *dest = new GlassBSDF(bsdf);
}

__host__
BSDF *GlassBSDF::copyToDev()
{
  BSDF **location;
  cudaCheckError( cudaMalloc(&location, sizeof(BSDF *)) );
  copyGlassBSDF<<<1, 1>>>(*this, location);
  cudaCheckError( cudaGetLastError() );
  cudaCheckError( cudaDeviceSynchronize() );
  BSDF *devLoc;
  cudaMemcpy(&devLoc, location, sizeof(BSDF *), cudaMemcpyDeviceToHost);
  return devLoc;
}


// Emission BSDF //

__device__
Spectrum EmissionBSDF::f(const Vector3D& wo, const Vector3D& wi) {
  return Spectrum();
}

__device__
Spectrum EmissionBSDF::sample_f(const Vector3D& wo, Vector3D* wi, float* pdf,
                                bool& inMat, curandState *state) {
  inMat = false;
  *wi  = sampler.get_sample(state, pdf);
  return Spectrum();
}

__global__ void copyEmissionBSDF(EmissionBSDF bsdf, BSDF **dest)
{
  *dest = new EmissionBSDF(bsdf);
  Spectrum r = (*dest)->get_emission();
}

__host__
BSDF *EmissionBSDF::copyToDev()
{
  BSDF **location;
  cudaCheckError( cudaMalloc(&location, sizeof(BSDF *)) );
  copyEmissionBSDF<<<1, 1>>>(*this, location);
  cudaCheckError( cudaGetLastError() );
  cudaCheckError( cudaDeviceSynchronize() );
  BSDF *devLoc;
  cudaMemcpy(&devLoc, location, sizeof(BSDF *), cudaMemcpyDeviceToHost);
  return devLoc;
}


} // namespace CMU462
