// Scene.h

#pragma once
#include "pathtracer.h"
#include "gpu/bvhGPU.h"

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif
#include <stdlib.h>
#include <GL/glew.h>

#include <glm/glm.hpp>



#include "IScene.h"
#include "ShaderWithVariables.h"

///@brief The Scene class renders everything in the VR world that will be the same
/// in the Oculus and Control windows. The RenderForOneEye function is the display entry point.
class Scene : public IScene
{
public:
    Scene();
    virtual ~Scene();

    virtual void initGL();
    virtual void timestep(double absTime, double dt);
    virtual void RenderForOneEye(const float* pMview, const float* pPersp) const;

    int sceneIdentifier() const { return 0; };

    CMU462::Application *app;

    void DrawScene(
        const glm::mat4& modelview,
        const glm::mat4& projection,
        const glm::mat4& object) const;

    void runStep(char*);
    void initCuda();

    GLuint fbo[2], tex[2];
    struct cudaGraphicsResource *cuda_fbo_resource[2];

    dim3 blockDim, gridDim;
    curandState *state;
    int stateSize;
 
    int eye;
    int eye_w, eye_h;
    int image_w, image_h;
    GLuint outFbo;

    VRRT::SceneLight *lights;
 
    cudaArray_t devRenderbuffer[2];
    unsigned char *devOutput;
  
    VRRT::BVHGPU bvh;



public:
    float m_amplitude;

private: // Disallow copy ctor and assignment operator
    Scene(const Scene&);
    Scene& operator=(const Scene&);

};
