// OVRSDK05AppSkeleton.h

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
class OVRSDK05AppSkeleton : public AppSkeleton
{
public:
    OVRSDK05AppSkeleton();
    virtual ~OVRSDK05AppSkeleton();

    void initHMD();
    void initVR(bool swapBackBufferDims = false);
    void exitVR();
    void RecenterPose();
    int ConfigureRendering();
    int ConfigureSDKRendering();
    int ConfigureClientRendering();

    void ToggleVignette();
    void ToggleTimeWarp();
    void ToggleOverdrive();
    void ToggleLowPersistence();
    void ToggleMirrorToWindow();
    void ToggleDynamicPrediction();

    void DismissHealthAndSafetyWarning() const;
    void CheckForTapToDismissHealthAndSafetyWarning() const;

    void display_stereo_undistorted() const;
    void display_sdk() const;
    void display_client() const;

    // Direct mode and SDK rendering hooks
    void AttachToWindow(void* pWindow) { ovrHmd_AttachToWindow(m_Hmd, pWindow, NULL, NULL); }
#if defined(OVR_OS_WIN32)
    void setWindow(HWND w) { m_Cfg.OGL.Window = w; }
#elif defined(OVR_OS_LINUX)
    void setWindow(_XDisplay* Disp) { m_Cfg.OGL.Disp = Disp; }
#endif

    virtual void timestep(double absTime, double dt);

    ovrSizei getHmdResolution() const;
    ovrVector2i getHmdWindowPos() const;
    bool UsingDebugHmd() const { return m_usingDebugHmd; }
    bool UsingDirectMode() const { return m_directHmdMode; }

    void _initPresentDistMesh(ShaderWithVariables& shader, int eyeIdx);
    virtual glm::ivec2 getRTSize() const;

    virtual glm::mat4 makeWorldToEyeMatrix() const;

    ovrHmd m_Hmd;
    ovrFovPort m_EyeFov[2];
    ovrGLConfig m_Cfg;
    ovrEyeRenderDesc m_EyeRenderDesc[2];
    ovrGLTexture m_EyeTexture[2];
    unsigned int m_hmdCaps;
    unsigned int m_distortionCaps;
    bool m_usingDebugHmd;
    bool m_directHmdMode;

    // For client rendering
    ovrRecti m_RenderViewports[2];
    ovrDistortionMesh m_DistMeshes[2];
    mutable ovrPosef m_eyePoseCached;

    char m_logUserData[256];

private: // Disallow copy ctor and assignment operator
    OVRSDK05AppSkeleton(const OVRSDK05AppSkeleton&);
    OVRSDK05AppSkeleton& operator=(const OVRSDK05AppSkeleton&);
};
