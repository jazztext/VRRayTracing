#ifndef __STATICSCENE_BSDF_H__
#define __STATICSCENE_BSDF_H__

#include "misc.h"
#include "spectrum.h"
#include "vector3D.h"
#include "matrix3x3.h"

#include "sampler.h"

#include <curand_kernel.h>

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
  virtual Spectrum f (const Vector3D& wo, const Vector3D& wi) = 0;

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
  virtual Spectrum sample_f (const Vector3D& wo, Vector3D* wi, float* pdf,
                             bool& inMat, curandState *state) = 0;

  /**
   * Get the emission value of the surface material. For non-emitting surfaces
   * this would be a zero energy spectrum.
   * \return emission spectrum of the surface material
   */
  __device__
  virtual Spectrum get_emission () const = 0;

  /**
   * If the BSDF is a delta distribution. Materials that are perfectly specular,
   * (e.g. water, glass, mirror) only scatter light from a single incident angle
   * to a single outgoing angle. These BSDFs are best described with alpha
   * distributions that are zero except for the single direction where light is
   * scattered.
   */
  __device__
  virtual bool is_delta() const = 0;

  /**
   * Reflection helper
   */
  __device__
  virtual void reflect(const Vector3D& wo, Vector3D* wi);

  /**
   * Refraction helper
   */
  __device__
  virtual bool refract(const Vector3D& wo, Vector3D* wi, float ior, bool inMat);

  __host__
  virtual BSDF *copyToDev() =0;

}; // class BSDF

/**
 * Diffuse BSDF.
 */
class DiffuseBSDF : public BSDF {
 public:

  __host__
  DiffuseBSDF(const CMU462::Spectrum& a) : albedo(a) { }
  __device__
  DiffuseBSDF(const Spectrum& a) : albedo(a) { }

  __device__
  Spectrum f(const Vector3D& wo, const Vector3D& wi);
  __device__
  Spectrum sample_f(const Vector3D& wo, Vector3D* wi, float* pdf, bool& inMat,
                    curandState *state);
  __device__
  Spectrum get_emission() const { return Spectrum(); }
  __device__
  bool is_delta() const { return false; }

  __host__
  BSDF *copyToDev();

private:

  Spectrum albedo;
  CosineWeightedHemisphereSampler3D sampler;

}; // class DiffuseBSDF

/**
 * Mirror BSDF
 */
class MirrorBSDF : public BSDF {
 public:

  __host__
  MirrorBSDF(const CMU462::Spectrum& reflectance) : reflectance(reflectance) { }
  __device__
  MirrorBSDF(const MirrorBSDF& bsdf) : reflectance(bsdf.reflectance) { }

  __device__
  Spectrum f(const Vector3D& wo, const Vector3D& wi);
  __device__
  Spectrum sample_f(const Vector3D& wo, Vector3D* wi, float* pdf, bool& inMat,
                    curandState *state);
  __device__
  Spectrum get_emission() const { return Spectrum(); }
  __device__
  bool is_delta() const { return true; }

  __host__
  BSDF *copyToDev();


private:

  float roughness;
  Spectrum reflectance;

}; // class MirrorBSDF*/

/**
 * Glossy BSDF.
 */
/*
class GlossyBSDF : public BSDF {
 public:

  GlossyBSDF(const Spectrum& reflectance, float roughness)
    : reflectance(reflectance), roughness(roughness) { }

  Spectrum f(const Vector3D& wo, const Vector3D& wi);
  Spectrum sample_f(const Vector3D& wo, Vector3D* wi, float* pdf);
  Spectrum get_emission() const { return Spectrum(); }
  bool is_delta() const { return false; }

private:

  float roughness;
  Spectrum reflectance;

}; // class GlossyBSDF*/

/**
 * Refraction BSDF.
 */
class RefractionBSDF : public BSDF {
 public:

  __host__
  RefractionBSDF(const CMU462::Spectrum& transmittance, float roughness,
                 float ior)
    : transmittance(transmittance), roughness(roughness), ior(ior) { }
  __device__
  RefractionBSDF(const RefractionBSDF& bsdf) : transmittance(bsdf.transmittance) ,
    roughness(bsdf.roughness), ior(bsdf.ior) { }

  __device__
  Spectrum f(const Vector3D& wo, const Vector3D& wi);
  __device__
  Spectrum sample_f(const Vector3D& wo, Vector3D* wi, float* pdf, bool& inMat,
                    curandState *state);
  __device__
  Spectrum get_emission() const { return Spectrum(); }
  __device__
  bool is_delta() const { return true; }

  __host__
  BSDF *copyToDev();


 private:

  float ior;
  float roughness;
  Spectrum transmittance;

}; // class RefractionBSDF

/**
 * Glass BSDF.
 */
class GlassBSDF : public BSDF {
 public:

  __host__
  GlassBSDF(const CMU462::Spectrum& transmittance, const Spectrum& reflectance,
            float roughness, float ior) :
    transmittance(transmittance), reflectance(reflectance),
    roughness(roughness), ior(ior) { }
  __device__
  GlassBSDF(const GlassBSDF& bsdf) : transmittance(bsdf.transmittance),
    reflectance(bsdf.reflectance), roughness(bsdf.roughness), ior(bsdf.ior) { }

  __device__
  Spectrum f(const Vector3D& wo, const Vector3D& wi);
  __device__
  Spectrum sample_f(const Vector3D& wo, Vector3D* wi, float* pdf, bool& inMat,
                    curandState *state);
  __device__
  Spectrum get_emission() const { return Spectrum(); }
  __device__
  bool is_delta() const { return true; }

  __host__
  BSDF *copyToDev();


 private:

  float ior;
  float roughness;
  Spectrum reflectance;
  Spectrum transmittance;

}; // class GlassBSDF

/**
 * Emission BSDF.
 */
class EmissionBSDF : public BSDF {
 public:

  __host__
  EmissionBSDF(const CMU462::Spectrum& radiance) : radiance(radiance) { }
  __device__
  EmissionBSDF(const EmissionBSDF& bsdf) : radiance(bsdf.radiance) { }

  __device__
  Spectrum f(const Vector3D& wo, const Vector3D& wi);
  __device__
  Spectrum sample_f(const Vector3D& wo, Vector3D* wi, float* pdf, bool& inMat,
                    curandState *state);
  __device__
  Spectrum get_emission() const { return radiance * (1.0 / PI); }
  __device__
  bool is_delta() const { return false; }

  __host__
  BSDF *copyToDev();


 private:

  Spectrum radiance;
  CosineWeightedHemisphereSampler3D sampler;

}; // class EmissionBSDF

}  // namespace CMU462

#endif  // CMU462_STATICSCENE_BSDF_H
