#include "light.h"

namespace VRRT {

// Directional Light //

__host__
DirectionalLight::DirectionalLight(const CMU462::Spectrum& rad,
                                   const CMU462::Vector3D& lightDir)
    : radiance(rad) {
  dirToLight = Vector3D(-lightDir.unit());
}

__device__
DirectionalLight::DirectionalLight(const Spectrum& rad,
                                   const Vector3D& lightDir)
    : radiance(rad) {
  dirToLight = -lightDir.unit();
}


__device__
Spectrum DirectionalLight::sample_L(const Vector3D& p, Vector3D* wi,
                                    float* distToLight, float* pdf,
                                    curandState *state) const {
  *wi = dirToLight;
  *distToLight = infy();
  *pdf = 1.0;
  return radiance;
}

__global__ void copyDirectionalLight(Spectrum rad, Vector3D lightDir,
                                     SceneLight **dest)
{
  *dest = new DirectionalLight(rad, lightDir);
}

__host__
void DirectionalLight::copyToDev(SceneLight **dest) {
  copyDirectionalLight<<<1, 1>>>(radiance, dirToLight, dest);
  cudaCheckError( cudaGetLastError() );
  cudaCheckError( cudaDeviceSynchronize() );
}

// Infinite Hemisphere Light //

__host__
InfiniteHemisphereLight::InfiniteHemisphereLight(const CMU462::Spectrum& rad)
    : radiance(rad) {
  CMU462::Matrix3x3 sampleToWorldH;
  sampleToWorldH[0] = CMU462::Vector3D(1,  0,  0);
  sampleToWorldH[1] = CMU462::Vector3D(0,  0, -1);
  sampleToWorldH[2] = CMU462::Vector3D(0,  1,  0);
  sampleToWorld = Matrix3x3(sampleToWorldH);
}

__device__
InfiniteHemisphereLight::InfiniteHemisphereLight(const Spectrum& rad)
    : radiance(rad) {
  sampleToWorld[0] = Vector3D(1,  0,  0);
  sampleToWorld[1] = Vector3D(0,  0, -1);
  sampleToWorld[2] = Vector3D(0,  1,  0);
}

__device__
Spectrum InfiniteHemisphereLight::sample_L(const Vector3D& p,
                                           Vector3D* wi,
                                           float* distToLight,
                                           float* pdf,
                                           curandState *state) const {
  Vector3D dir = sampler.get_sample(state);
  *wi = sampleToWorld* dir;
  *distToLight = infy();
  *pdf = 1.0 / (2.0 * M_PI);
  return radiance;
}

__global__ void copyInfiniteHemisphereLight(Spectrum rad, SceneLight **dest)
{
  *dest = new InfiniteHemisphereLight(rad);
}

__host__
void InfiniteHemisphereLight::copyToDev(SceneLight **dest) {
  copyInfiniteHemisphereLight<<<1, 1>>>(radiance, dest);
  cudaCheckError( cudaGetLastError() );
  cudaCheckError( cudaDeviceSynchronize() );
}

// Point Light //

__host__
PointLight::PointLight(const CMU462::Spectrum& rad,
                       const CMU462::Vector3D& pos) :
  radiance(rad), position(pos) { }

__device__
PointLight::PointLight(const Spectrum& rad, const Vector3D& pos) :
  radiance(rad), position(pos) { }

__device__
Spectrum PointLight::sample_L(const Vector3D& p, Vector3D* wi,
                             float* distToLight,
                             float* pdf, curandState *state) const {
  Vector3D d = position - p;
  *wi = d.unit();
  *distToLight = d.norm();
  *pdf = 1.0;
  return radiance;
}

__global__ void copyPointLight(Spectrum rad, Vector3D pos, SceneLight **dest)
{
  *dest = new PointLight(rad, pos);
}

__host__
void PointLight::copyToDev(SceneLight **dest) {
  copyPointLight<<<1, 1>>>(radiance, position, dest);
  cudaCheckError( cudaGetLastError() );
  cudaCheckError( cudaDeviceSynchronize() );
}

// Spot Light //

__host__
SpotLight::SpotLight(const CMU462::Spectrum& rad, const CMU462::Vector3D& pos,
                     const CMU462::Vector3D& dir, float angle) {

}

__device__
SpotLight::SpotLight(const Spectrum& rad, const Vector3D& pos,
                     const Vector3D& dir, float angle) {

}

__device__
Spectrum SpotLight::sample_L(const Vector3D& p, Vector3D* wi,
                             float* distToLight, float* pdf,
                             curandState *state) const {
  return Spectrum();
}

__global__ void copySpotLight(Spectrum rad, Vector3D pos, Vector3D dir,
                              float angle, SceneLight **dest)
{
  *dest = new SpotLight(rad, pos, dir, angle);
}

__host__
void SpotLight::copyToDev(SceneLight **dest) {
  copySpotLight<<<1, 1>>>(radiance, position, direction, angle, dest);
  cudaCheckError( cudaGetLastError() );
  cudaCheckError( cudaDeviceSynchronize() );
}

// Area Light //

__host__
AreaLight::AreaLight(const CMU462::Spectrum& rad,
                     const CMU462::Vector3D& pos,
                     const CMU462::Vector3D& dir,
                     const CMU462::Vector3D& dim_x,
                     const CMU462::Vector3D& dim_y)
  : radiance(rad), position(pos), direction(dir),
    dim_x(dim_x), dim_y(dim_y), area(dim_x.norm() * dim_y.norm()) { }

__device__
AreaLight::AreaLight(const Spectrum& rad,
                     const Vector3D& pos,   const Vector3D& dir,
                     const Vector3D& dim_x, const Vector3D& dim_y)
  : radiance(rad), position(pos), direction(dir),
    dim_x(dim_x), dim_y(dim_y), area(dim_x.norm() * dim_y.norm()) { }

__device__
Spectrum AreaLight::sample_L(const Vector3D& p, Vector3D* wi,
                             float* distToLight, float* pdf,
                             curandState *state) const {

  Vector2D sample = sampler.get_sample(state) - Vector2D(0.5f, 0.5f);
  Vector3D d = position + sample.x * dim_x + sample.y * dim_y - p;
  float cosTheta = dot(d, direction);
  float sqDist = d.norm2();
  float dist = sqrtf(sqDist);
  *wi = d / dist;
  *distToLight = dist - .01;
  *pdf = sqDist / (area * fabsf(cosTheta));
  return cosTheta < 0 ? radiance : Spectrum();
};

__global__ void copyAreaLight(Spectrum rad, Vector3D pos, Vector3D dir,
                              Vector3D dim_x, Vector3D dim_y, SceneLight **dest)
{
  *dest = new AreaLight(rad, pos, dir, dim_x, dim_y);
}

__host__
void AreaLight::copyToDev(SceneLight **dest) {
  copyAreaLight<<<1, 1>>>(radiance, position, direction, dim_x, dim_y, dest);
  cudaCheckError( cudaGetLastError() );
  cudaCheckError( cudaDeviceSynchronize() );
}

} // namespace StaticScene
