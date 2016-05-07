#ifndef __SPECTRUM_H__
#define __SPECTRUM_H__

#include "CMU462/spectrum.h"

namespace VRRT {

/**
 * Encodes radiance & irradiance values by the intensity of each visible
 * spectrum. Note that this is not strictly an actual spectrum with all
 * wavelengths, but it gives us enough information as we can only sense
 * a particular wavelengths.
 */
class Spectrum {
 public:
  float r;  ///< intensity of red spectrum
  float g;  ///< intensity of green spectrum
  float b;  ///< intensity of blue spectrum

  /**
   * Parameterized Constructor.
   * Initialize from component values.
   * \param r Intensity of the red spectrum
   * \param g Intensity of the green spectrum
   * \param b Intensity of the blue spectrum
   */
  __device__
  static Spectrum make(float r_ = 0, float g_ = 0, float b_ = 0)
  {
    Spectrum s;
    s.r = r_;
    s.g = g_;
    s.b = b_;
    return s;
  }

  __host__
  static Spectrum make(const CMU462::Spectrum &s)
  {
    Spectrum s1;
    s1.r = s.r;
    s1.g = s.g;
    s1.b = s.b;
    return s1;
  }

  // operators //

  __device__
  inline Spectrum operator+(const Spectrum &rhs) const {
    return Spectrum::make(r + rhs.r, g + rhs.g, b + rhs.b);
  }

  __device__
  inline Spectrum &operator+=(const Spectrum &rhs) {
    r += rhs.r;
    g += rhs.g;
    b += rhs.b;
    return *this;
  }

  __device__
  inline Spectrum operator*(const Spectrum &rhs) const {
    return Spectrum::make(r * rhs.r, g * rhs.g, b * rhs.b);
  }

  __device__
  inline Spectrum &operator*=(const Spectrum &rhs) {
    r *= rhs.r;
    g *= rhs.g;
    b *= rhs.b;
    return *this;
  }

  __device__
  inline Spectrum operator*(float s) const {
    return Spectrum::make(r * s, g * s, b * s);
  }

  __device__
  inline Spectrum &operator*=(float s) {
    r *= s;
    g *= s;
    b *= s;
    return *this;
  }

  __device__
  inline bool operator==(const Spectrum &rhs) const {
    return r == rhs.r && g == rhs.g && b == rhs.b;
  }

  __device__
  inline bool operator!=(const Spectrum &rhs) const {
    return !operator==(rhs);
  }

  __device__
  inline float illum() const {
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
  }

};  // class Spectrum

// Commutable scalar multiplication
__device__
inline Spectrum operator*(float s, const Spectrum &c) { return c * s; }

}  // namespace CMU462

#endif  // CMU462_SPECTRUM_H
