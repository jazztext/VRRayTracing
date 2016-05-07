#ifndef __STATICSCENE_BSDF_H__
#define __STATICSCENE_BSDF_H__

#include "misc.h"
#include "spectrum.h"
#include "vector3D.h"
#include "matrix3x3.h"

#include "sampler.h"

#include <curand_kernel.h>
#include <iostream>

namespace VRRT {

// Helper math functions. Assume all vectors are in unit hemisphere //

__device__ inline float clamp (float n, float lower, float upper) {
  return fmaxf(lower, fminf(n, upper));
}

__device__ inline float cos_theta(const Vector3D& w) {
  return w.z;
}

__device__ inline float abs_cos_theta(const Vector3D& w) {
  return fabsf(w.z);
}

__device__ inline float sin_theta2(const Vector3D& w) {
  return fmaxf(0.0, 1.0 - cos_theta(w) * cos_theta(w));
}

__device__ inline float sin_theta(const Vector3D& w) {
  return sqrtf(sin_theta2(w));
}

__device__ inline float cos_phi(const Vector3D& w) {
  float sinTheta = sin_theta(w);
  if (sinTheta == 0.0) return 1.0;
  return clamp(w.x / sinTheta, -1.0, 1.0);
}

__device__ inline float sin_phi(const Vector3D& w) {
  float sinTheta = sin_theta(w);
  if (sinTheta) return 0.0;
  return clamp(w.y / sinTheta, -1.0, 1.0);
}

__device__ void make_coord_space(Matrix3x3& o2w, const Vector3D& n);

/**
 * Interface for BSDFs.
 */
class BSDF {
 public:
   enum BSDFType { DIFFUSE, MIRROR, GLASS, EMISSION };

  __host__
  BSDF(const CMU462::Spectrum& c, BSDFType t) : color(c), t(t)
  {
    if (t != DIFFUSE && t != MIRROR && t != EMISSION)
      std::cout << "BSDF construction error\n";
  }
  __host__
  BSDF(const CMU462::Spectrum& transmittance, const Spectrum& reflectance,
       float roughness, float ior, BSDFType t) :
    color(transmittance), color2(reflectance), roughness(roughness), ior(ior),
    t(t)
  {
    if (t != GLASS) std::cout << "BSDF construction error\n";
  }


  /**
   * Evaluate BSDF.
   * Given incident light direction wi and outgoing light direction wo. Note
   * that both wi and wo are defined in the local coordinate system at the
   * point of intersection.
   * \param wo outgoing light direction in local space of point of intersection
   * \param wi incident light direction in local space of point of intersection
   * \return reflectance in the given incident/outgoing directions
   */
  __device__
  Spectrum f (const Vector3D& wo, const Vector3D& wi);

  /**
   * Evaluate BSDF.
   * Given the outgoing light direction wo, compute the incident light
   * direction and store it in wi. Store the pdf of the outgoing light in pdf.
   * Again, note that wo and wi should both be defined in the local coordinate
   * system at the point of intersection.
   * \param wo outgoing light direction in local space of point of intersection
   * \param wi address to store incident light direction
   * \param pdf address to store the pdf of the output incident direction
   * \return reflectance in the output incident and given outgoing directions
   */
  __device__
  Spectrum sample_f (const Vector3D& wo, Vector3D* wi, float* pdf,
                     bool& inMat, curandState *state);

  __device__
  Spectrum glassSample(const Vector3D& wo, Vector3D* wi, float* pdf,
                       bool& inMat, curandState *state);

  /**
   * Get the emission value of the surface material. For non-emitting surfaces
   * this would be a zero energy spectrum.
   * \return emission spectrum of the surface material
   */
  __device__
  Spectrum get_emission ();

  /**
   * If the BSDF is a delta distribution. Materials that are perfectly specular,
   * (e.g. water, glass, mirror) only scatter light from a single incident angle
   * to a single outgoing angle. These BSDFs are best described with alpha
   * distributions that are zero except for the single direction where light is
   * scattered.
   */
  __device__
  bool is_delta();

  /**
   * Reflection helper
   */
  __device__
  void reflect(const Vector3D& wo, Vector3D* wi);

  /**
   * Refraction helper
   */
  __device__
  bool refract(const Vector3D& wo, Vector3D* wi, float ior, bool inMat);

  __host__
  BSDF *copyToDev();

 private:
  BSDFType t;
  Spectrum color, color2; //can be albedo, reflectance, transmittance, etc.
  CosineWeightedHemisphereSampler3D sampler;
  float roughness;
  float ior;


}; // class BSDF

}  // namespace VRRT

#endif  // CMU462_STATICSCENE_BSDF_H
