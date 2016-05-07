// OVRSDK07AppSkeleton.h

#pragma once

#ifdef __APPLE__
#include "opengl/gl.h"
#endif

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

#include "AppSkeleton.h"

#include <Kernel/OVR_Types.h> // Pull in OVR_OS_* defines 
#include <OVR_CAPI.h>
#include <OVR_CAPI_GL.h>

///@brief Encapsulates as much of the VR viewer state as possible,
/// pushing all viewer-independent stuff to Scene.
class OVRSDK07AppSkeleton : public AppSkeleton
{
public:
    enum MirrorType {
        MirrorNone = 0,
        MirrorDistorted,
        MirrorUndistorted,
        NumMirrorTypes
    };

    OVRSDK07AppSkeleton();
    virtual ~OVRSDK07AppSkeleton();

    void initHMD();
    void initVR(bool swapBackBufferDims = false);
    void RecenterPose();
    void exitVR();

    void ToggleVignette() {}
    void ToggleTimeWarp() {}
    void ToggleOverdrive() {}
    void ToggleLowPersistence() {}
    void ToggleDynamicPrediction() {}
    void ToggleMirroringType();
    void ToggleQuadInWorld() { m_showQuadInWorld = !m_showQuadInWorld; }

    void SetAppWindowSize(ovrSizei sz) { m_appWindowSize = sz; }

    ovrSizei getHmdResolution() const;

    void display_stereo_undistorted() { display_sdk(); }
    void display_sdk() const;
    void display_client() const { display_sdk(); }

protected:
    void BlitLeftEyeRenderToUndistortedMirrorTexture() const;

    ovrHmd m_Hmd;
    ovrEyeRenderDesc m_eyeRenderDescs[ovrEye_Count];
    ovrVector3f m_eyeOffsets[ovrEye_Count];
    glm::mat4 m_eyeProjections[ovrEye_Count];

    mutable ovrPosef m_eyePoses[ovrEye_Count];
    mutable ovrLayerEyeFov m_layerEyeFov;
    mutable ovrLayerQuad m_layerQuad;
    mutable int m_frameIndex;

    ovrTexture* m_pMirrorTex;
    ovrSwapTextureSet* m_pTexSet[ovrEye_Count];
    ovrSwapTextureSet* m_pQuadTex;
    FBO m_swapFBO;
    FBO m_quadFBO;
    FBO m_mirrorFBO;
    FBO m_undistortedFBO;

    ovrSizei m_appWindowSize;
    MirrorType m_mirror;
    bool m_showQuadInWorld;

public:
    glm::vec3 m_quadLocation;

private: // Disallow copy ctor and assignment operator
    OVRSDK07AppSkeleton(const OVRSDK07AppSkeleton&);
    OVRSDK07AppSkeleton& operator=(const OVRSDK07AppSkeleton&);
};
