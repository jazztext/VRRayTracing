// OVRSDK08AppSkeleton.cpp
// Huge thanks to jherico
// https://github.com/jherico/OculusRiftInAction/blob/ea0231ad045187c8a5819a801ce4a2fae63301aa/examples/cpp/Example_SelfContained.cpp

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif
#include <GL/glew.h>

#ifdef USE_ANTTWEAKBAR
#  include <AntTweakBar.h>
#endif

#define _USE_MATH_DEFINES
#include <math.h>
#include <algorithm>
#include <iostream>
#include <fstream>

#include <OVR.h>

#include "OVRSDK08AppSkeleton.h"
#include "ShaderFunctions.h"
#include "MatrixFunctions.h"
#include "GLUtils.h"
#include "Logger.h"

#define USE_OVR_PERF_LOGGING

#ifndef PROJECT_NAME
// This macro should be defined in CMakeLists.txt
#define PROJECT_NAME "RiftSkeleton"
#endif

OVRSDK08AppSkeleton::OVRSDK08AppSkeleton()
: m_Hmd(NULL)
, m_frameIndex(0)
, m_pMirrorTex(NULL)
, m_mirror(MirrorDistorted)
, m_perfHudMode(ovrPerfHud_Off)
{
    m_pTexSet[0] = NULL;
    m_pTexSet[1] = NULL;

    memset((void*)&m_tweakbarQuad, 0, sizeof(worldQuad));
    m_tweakbarQuad.m_showQuadInWorld = false;
    m_tweakbarQuad.m_quadLocation = glm::vec3(-.5f, .3f, -1.f);
    m_tweakbarQuad.m_quadRotation = glm::vec3(0.f);

    memset((void*)&m_secondQuad, 0, sizeof(worldQuad));
    m_secondQuad.m_showQuadInWorld = false;
    m_secondQuad.m_quadLocation = glm::vec3(.5f, .3f, -1.f);
    m_secondQuad.m_quadRotation = glm::vec3(0.f);
}

OVRSDK08AppSkeleton::~OVRSDK08AppSkeleton()
{
}

void OVRSDK08AppSkeleton::RecenterPose()
{
    if (m_Hmd == NULL)
        return;
    ovr_RecenterPose(m_Hmd);
}

void OVRSDK08AppSkeleton::OnMouseButton(int button, int action)
{
    // Tweakbar clicks are handled in main with
    // windowing-framework-specific calls that alter Ant state.
}

void OVRSDK08AppSkeleton::OnMouseMove(int x, int y)
{
    const FBO& f = m_tweakbarQuad.fbo;
    m_pointerCoords.x = static_cast<float>(x) / static_cast<float>(f.w);
    m_pointerCoords.y = static_cast<float>(y) / static_cast<float>(f.h);
}

void OVRSDK08AppSkeleton::OnMouseWheel(double xd, double yd)
{
}

void OVRSDK08AppSkeleton::ToggleMirroringType()
{
    int m = static_cast<int>(m_mirror);
    ++m %= NumMirrorTypes;
    m_mirror = static_cast<MirrorType>(m);
}

void OVRSDK08AppSkeleton::TogglePerfHud()
{
    int phm = static_cast<int>(m_perfHudMode);
    ++phm %= static_cast<int>(ovrPerfHud_Count);
    m_perfHudMode = static_cast<ovrPerfHudMode>(phm);
    ovr_SetInt(m_Hmd, OVR_PERF_HUD_MODE, m_perfHudMode);
}

void OVRSDK08AppSkeleton::ToggleQuadInWorld()
{
    m_tweakbarQuad.m_showQuadInWorld = !m_tweakbarQuad.m_showQuadInWorld;
    m_secondQuad.m_showQuadInWorld = m_tweakbarQuad.m_showQuadInWorld;
    if (m_tweakbarQuad.m_showQuadInWorld)
    {
        const FBO& f = m_tweakbarQuad.fbo;
#ifdef USE_ANTTWEAKBAR
        TwWindowSize(f.w, f.h);
#endif
    }
}

void OVRSDK08AppSkeleton::exitVR()
{
    _DestroySwapTextures();
    ovr_Destroy(m_Hmd);
    ovr_Shutdown();
}

void OVRSDK08AppSkeleton::_DestroySwapTextures()
{
    for (int i = 0; i < 2; ++i)
    {
        if (m_pTexSet[i])
        {
            ovr_DestroySwapTextureSet(m_Hmd, m_pTexSet[i]);
            m_pTexSet[i] = nullptr;
        }
    }
    ovr_DestroySwapTextureSet(m_Hmd, m_tweakbarQuad.m_pQuadTex);
    m_tweakbarQuad.m_pQuadTex = nullptr;
    ovr_DestroySwapTextureSet(m_Hmd, m_secondQuad.m_pQuadTex);
    m_secondQuad.m_pQuadTex = nullptr;
}

void OVRSDK08AppSkeleton::initHMD()
{
    const ovrResult res = ovr_Initialize(NULL);
    if (ovrSuccess != res)
    {
        LOG_ERROR("ovr_Initialize failed with code %d", res);
    }

    ovrGraphicsLuid luid;
    if (ovrSuccess != ovr_Create(&m_Hmd, &luid))
    {
        LOG_ERROR("ovr_Create failed with code %d", res);
    }

    const ovrResult ret = ovr_ConfigureTracking(m_Hmd,
        ovrTrackingCap_Orientation | ovrTrackingCap_MagYawCorrection | ovrTrackingCap_Position,
        ovrTrackingCap_Orientation);
    if (!OVR_SUCCESS(ret))
    {
        LOG_ERROR("Error calling ovr_ConfigureTracking");
    }
}

///@brief Called once a GL context has been set up.
void OVRSDK08AppSkeleton::initVR(bool swapBackBufferDims)
{
    if (m_Hmd == NULL)
        return;

    // Set up eye view parameters
    for (ovrEyeType eye = ovrEyeType::ovrEye_Left;
        eye < ovrEyeType::ovrEye_Count;
        eye = static_cast<ovrEyeType>(eye + 1))
    {
        ovrEyeRenderDesc& erd = m_eyeRenderDescs[eye];
        const ovrHmdDesc& hmd = ovr_GetHmdDesc(m_Hmd);
        erd = ovr_GetRenderDesc(m_Hmd, eye, hmd.MaxEyeFov[eye]);

        m_eyeOffsets[eye] = erd.HmdToEyeViewOffset;
        const ovrMatrix4f ovrPerspectiveProjection = ovrMatrix4f_Projection(
            erd.Fov, .1f, 10000.f, ovrProjection_RightHanded);
        m_eyeProjections[eye] = glm::transpose(glm::make_mat4(&ovrPerspectiveProjection.M[0][0]));
    }

    _DestroySwapTextures();

    // Create eye render target textures and FBOs
    ovrLayerEyeFov& layer = m_layerEyeFov;
    layer.Header.Type = ovrLayerType_EyeFov;
    layer.Header.Flags = ovrLayerFlag_TextureOriginAtBottomLeft;

    for (ovrEyeType eye = ovrEyeType::ovrEye_Left;
        eye < ovrEyeType::ovrEye_Count;
        eye = static_cast<ovrEyeType>(eye + 1))
    {
        const ovrHmdDesc& hmd = ovr_GetHmdDesc(m_Hmd);
        const ovrFovPort& fov = layer.Fov[eye] = hmd.MaxEyeFov[eye];
        const ovrSizei& size = layer.Viewport[eye].Size = ovr_GetFovTextureSize(m_Hmd, eye, fov, 1.f);
        layer.Viewport[eye].Pos = { 0, 0 };
        LOG_INFO("Eye %d tex : %dx%d @ (%d,%d)", eye, size.w, size.h,
            layer.Viewport[eye].Pos.x, layer.Viewport[eye].Pos.y);

        // Allocate the frameBuffer that will hold the scene, and then be
        // re-rendered to the screen with distortion
        if (!OVR_SUCCESS(ovr_CreateSwapTextureSetGL(m_Hmd, GL_RGBA, size.w, size.h, &m_pTexSet[eye])))
        {
            LOG_ERROR("Unable to create swap textures");
            return;
        }
        const ovrSwapTextureSet& swapSet = *m_pTexSet[eye];
        for (int i = 0; i < swapSet.TextureCount; ++i)
        {
            const ovrGLTexture& ovrTex = (ovrGLTexture&)swapSet.Textures[i];
            glBindTexture(GL_TEXTURE_2D, ovrTex.OGL.TexId);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        }

        // Manually assemble swap FBO
        m_swapFBO.w = size.w;
        m_swapFBO.h = size.h;
        glGenFramebuffers(1, &m_swapFBO.id);
        glBindFramebuffer(GL_FRAMEBUFFER, m_swapFBO.id);
        const int idx = 0;
        const ovrGLTextureData* pGLData = reinterpret_cast<ovrGLTextureData*>(&swapSet.Textures[idx]);
        m_swapFBO.tex = pGLData->TexId;
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_swapFBO.tex, 0);

        m_swapFBO.depth = 0;
        glGenRenderbuffers(1, &m_swapFBO.depth);
        glBindRenderbuffer(GL_RENDERBUFFER, m_swapFBO.depth);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT16, size.w, size.h);
        glBindRenderbuffer(GL_RENDERBUFFER, 0);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, m_swapFBO.depth);

        // Check status
        const GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
        if (status != GL_FRAMEBUFFER_COMPLETE)
        {
            LOG_ERROR("Framebuffer status incomplete: %d %x", status, status);
        }
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        layer.ColorTexture[eye] = m_pTexSet[eye];
    }

    // In-world quads
    const ovrSizei qsz = { 600, 600 };
    _InitQuadLayer(m_tweakbarQuad, qsz);
    _InitQuadLayer(m_secondQuad, qsz);

    m_cursorShader.initProgram("basic");
    m_cursorShader.bindVAO();
    _InitPointerAttributes();
    glBindVertexArray(0);

    // Mirror texture for displaying to desktop window
    if (m_pMirrorTex)
    {
        ovr_DestroyMirrorTexture(m_Hmd, m_pMirrorTex);
    }

    const ovrEyeType eye = ovrEyeType::ovrEye_Left;
    const ovrHmdDesc& hmd = ovr_GetHmdDesc(m_Hmd);
    const ovrFovPort& fov = layer.Fov[eye] = hmd.MaxEyeFov[eye];
    const ovrSizei& size = layer.Viewport[eye].Size = ovr_GetFovTextureSize(m_Hmd, eye, fov, 1.f);
    const ovrResult result = ovr_CreateMirrorTextureGL(m_Hmd, GL_RGBA, size.w, size.h, &m_pMirrorTex);
    if (!OVR_SUCCESS(result))
    {
        LOG_ERROR("Unable to create mirror texture");
        return;
    }

    // Manually assemble mirror FBO
    m_mirrorFBO.w = size.w;
    m_mirrorFBO.h = size.h;
    glGenFramebuffers(1, &m_mirrorFBO.id);
    glBindFramebuffer(GL_FRAMEBUFFER, m_mirrorFBO.id);
    const ovrGLTextureData* pMirrorGLData = reinterpret_cast<ovrGLTextureData*>(m_pMirrorTex);
    m_mirrorFBO.tex = pMirrorGLData->TexId;
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_mirrorFBO.tex, 0);

    // Check status
    {
        const GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
        if (status != GL_FRAMEBUFFER_COMPLETE)
        {
            LOG_ERROR("Framebuffer status incomplete: %d %x", status, status);
        }
    }

    // Create another FBO for blitting the undistorted scene to for desktop window display.
    m_undistortedFBO.w = size.w;
    m_undistortedFBO.h = size.h;
    glGenFramebuffers(1, &m_undistortedFBO.id);
    glBindFramebuffer(GL_FRAMEBUFFER, m_undistortedFBO.id);
    glGenTextures(1, &m_undistortedFBO.tex);
    glBindTexture(GL_TEXTURE_2D, m_undistortedFBO.tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8,
        m_undistortedFBO.w, m_undistortedFBO.h, 0,
        GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_undistortedFBO.tex, 0);

    // Check status
    {
        const GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
        if (status != GL_FRAMEBUFFER_COMPLETE)
        {
            LOG_ERROR("Framebuffer status incomplete: %d %x", status, status);
        }
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);

    glEnable(GL_DEPTH_TEST);
}

// Create in-world quad layer for display of a 2d texture.
void OVRSDK08AppSkeleton::_InitQuadLayer(
    worldQuad& quad,
    const ovrSizei size)
{
    if (!OVR_SUCCESS(ovr_CreateSwapTextureSetGL(m_Hmd, GL_RGBA, size.w, size.h, &quad.m_pQuadTex)))
    {
        LOG_ERROR("Unable to create quad layer swap tex");
        return;
    }

    const ovrSwapTextureSet& swapSet = *quad.m_pQuadTex;
    const ovrGLTexture& ovrTex = (ovrGLTexture&)swapSet.Textures[0];
    glBindTexture(GL_TEXTURE_2D, ovrTex.OGL.TexId);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    ovrLayerQuad& layer = quad.m_layerQuad;
    layer.Header.Type = ovrLayerType_Quad;
    layer.Header.Flags = ovrLayerFlag_TextureOriginAtBottomLeft;
    layer.ColorTexture = quad.m_pQuadTex;
    layer.Viewport.Pos = { 0, 0 };
    layer.Viewport.Size = size;
    layer.QuadPoseCenter.Orientation = { 0.f, 0.f, 0.f, 1.f };
    layer.QuadPoseCenter.Position = { .5f, .5f, .5f };
    layer.QuadSize = { 1.f, 1.f };

    // Manually assemble quad FBO
    quad.fbo.w = size.w;
    quad.fbo.h = size.h;
    glGenFramebuffers(1, &quad.fbo.id);
    glBindFramebuffer(GL_FRAMEBUFFER, quad.fbo.id);
    const int idx = 0;
    const ovrGLTextureData* pGLData = reinterpret_cast<ovrGLTextureData*>(&swapSet.Textures[0]);
    quad.fbo.tex = pGLData->TexId;
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, quad.fbo.tex, 0);

    quad.fbo.depth = 0;
    glGenRenderbuffers(1, &quad.fbo.depth);
    glBindRenderbuffer(GL_RENDERBUFFER, quad.fbo.depth);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT16, size.w, size.h);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, quad.fbo.depth);

    // Check status
    const GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE)
    {
        LOG_ERROR("Framebuffer status incomplete: %d %x", status, status);
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void OVRSDK08AppSkeleton::_InitPointerAttributes()
{
    const glm::vec3 verts[] = {
        glm::vec3(0.f),
        glm::vec3(.05f, .025f, .0f),
        glm::vec3(.025f, .05f, .0f),
    };

    const glm::vec3 cols[] = {
        glm::vec3(1.f, 0.f, 0.f),
        glm::vec3(0.f, 1.f, 0.f),
        glm::vec3(0.f, 0.f, 1.f),
    };

    GLuint vertVbo = 0;
    glGenBuffers(1, &vertVbo);
    m_cursorShader.AddVbo("vPosition", vertVbo);
    glBindBuffer(GL_ARRAY_BUFFER, vertVbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
    glVertexAttribPointer(m_cursorShader.GetAttrLoc("vPosition"), 3, GL_FLOAT, GL_FALSE, 0, NULL);

    GLuint colVbo = 0;
    glGenBuffers(1, &colVbo);
    m_cursorShader.AddVbo("vColor", colVbo);
    glBindBuffer(GL_ARRAY_BUFFER, colVbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(cols), cols, GL_STATIC_DRAW);
    glVertexAttribPointer(m_cursorShader.GetAttrLoc("vColor"), 3, GL_FLOAT, GL_FALSE, 0, NULL);

    glEnableVertexAttribArray(m_cursorShader.GetAttrLoc("vPosition"));
    glEnableVertexAttribArray(m_cursorShader.GetAttrLoc("vColor"));
}

ovrSizei OVRSDK08AppSkeleton::getHmdResolution() const
{
    if (m_Hmd == NULL)
    {
        return { 800, 600 };
    }
    const ovrHmdDesc& hmd = ovr_GetHmdDesc(m_Hmd);
    return hmd.Resolution;
}

void OVRSDK08AppSkeleton::BlitLeftEyeRenderToUndistortedMirrorTexture() const
{
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, m_swapFBO.id);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, m_undistortedFBO.id);
    glViewport(0, 0, m_undistortedFBO.w, m_undistortedFBO.h);
    glBlitFramebuffer(
        0, static_cast<int>(static_cast<float>(m_swapFBO.h)*m_fboScale),
        static_cast<int>(static_cast<float>(m_swapFBO.w)*m_fboScale), 0, ///@todo Fix for FBO scaling
        0, 0, m_undistortedFBO.w, m_undistortedFBO.h,
        GL_COLOR_BUFFER_BIT, GL_NEAREST);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, m_swapFBO.id);
}

// Draw to in-world quad
void OVRSDK08AppSkeleton::_DrawToTweakbarQuad() const
{
    const ovrSwapTextureSet& swapSet = *m_tweakbarQuad.m_pQuadTex;
    const FBO& f = m_tweakbarQuad.fbo;
    glBindFramebuffer(GL_FRAMEBUFFER, f.id);
    const ovrGLTexture& tex = (ovrGLTexture&)(swapSet.Textures[swapSet.CurrentIndex]);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex.OGL.TexId, 0);

    glViewport(0, 0, f.w, f.h);
    const float lum = .1f;
    glClearColor(lum, lum, lum, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
#ifdef USE_ANTTWEAKBAR
    TwDraw();
#endif

    _DrawCursor();

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void OVRSDK08AppSkeleton::_DrawCursor() const
{
    // Restore current program when we're done; we are rendering to FBO
    GLint prog = 0;
    glGetIntegerv(GL_CURRENT_PROGRAM, &prog);

    glUseProgram(m_cursorShader.prog());
    {
        const glm::mat4 modelview = glm::translate(glm::mat4(1.f), glm::vec3(m_pointerCoords, 0.f));
        const glm::mat4 projection = glm::ortho(0.f, 1.f, 1.f, 0.f, -1.f, 1.f);

        glUniformMatrix4fv(m_cursorShader.GetUniLoc("mvmtx"), 1, false, glm::value_ptr(modelview));
        glUniformMatrix4fv(m_cursorShader.GetUniLoc("prmtx"), 1, false, glm::value_ptr(projection));

        m_cursorShader.bindVAO();
        {
            glDrawArrays(GL_TRIANGLE_FAN, 0, 3);
        }
        glBindVertexArray(0);
    }
    glUseProgram(prog);
}

void OVRSDK08AppSkeleton::_DrawToSecondQuad() const
{
    const ovrSwapTextureSet& swapSet = *m_secondQuad.m_pQuadTex;
    glBindFramebuffer(GL_FRAMEBUFFER, m_secondQuad.fbo.id);
    const ovrGLTexture& tex = (ovrGLTexture&)(swapSet.Textures[swapSet.CurrentIndex]);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex.OGL.TexId, 0);

    const float lum = .1f;
    glClearColor(lum, .8f, lum, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void OVRSDK08AppSkeleton::display_sdk() const
{
    const ovrHmd hmd = m_Hmd;
    if (hmd == NULL)
    {
        display_buffered();
        return;
    }

    ovrTrackingState outHmdTrackingState = { 0 };
    ovr_GetEyePoses(hmd, m_frameIndex, false, m_eyeOffsets,
        m_eyePoses, &outHmdTrackingState);

    for (ovrEyeType eye = ovrEyeType::ovrEye_Left;
        eye < ovrEyeType::ovrEye_Count;
        eye = static_cast<ovrEyeType>(eye + 1))
    {
        const ovrSwapTextureSet& swapSet = *m_pTexSet[eye];
        glBindFramebuffer(GL_FRAMEBUFFER, m_swapFBO.id);
        const ovrGLTexture& tex = (ovrGLTexture&)(swapSet.Textures[swapSet.CurrentIndex]);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex.OGL.TexId, 0);
        {
            // Handle render target resolution scaling
            m_layerEyeFov.Viewport[eye].Size = ovr_GetFovTextureSize(m_Hmd, eye, m_layerEyeFov.Fov[eye], m_fboScale);
            ovrRecti& vp = m_layerEyeFov.Viewport[eye];
            if (m_layerEyeFov.Header.Flags & ovrLayerFlag_TextureOriginAtBottomLeft)
            {
                ///@note It seems that the render viewport should be vertically centered within the swapSet texture.
                /// See also OculusWorldDemo.cpp:1443 - "The usual OpenGL viewports-don't-match-UVs oddness."
                const int texh = swapSet.Textures[swapSet.CurrentIndex].Header.TextureSize.h;
                vp.Pos.y = (texh - vp.Size.h) / 2;
            }

            glViewport(vp.Pos.x, vp.Pos.y, vp.Size.w, vp.Size.h);


            glClearColor(0.f, 0.f, 0.f, 0.f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


            // Cinemascope - letterbox bars scissoring off pixels above and below vp center
            const float hc = .5f * m_cinemaScope;
            const int scisPx = static_cast<int>(hc * static_cast<float>(vp.Size.h));
            ovrRecti sp(vp);
            sp.Pos.y += scisPx;
            sp.Size.h -= 2 * scisPx;
            glScissor(sp.Pos.x, sp.Pos.y, sp.Size.w, sp.Size.h);
            glEnable(GL_SCISSOR_TEST);


            // Render the scene for the current eye
            const ovrPosef& eyePose = m_eyePoses[eye];
            const glm::mat4 viewLocal = makeMatrixFromPose(eyePose);
            const glm::mat4 viewWorld = makeWorldToChassisMatrix() * viewLocal;
            const glm::mat4& proj = m_eyeProjections[eye];
            _DrawScenes(
                glm::value_ptr(glm::inverse(viewWorld)),
                glm::value_ptr(proj),
                glm::value_ptr(glm::inverse(viewLocal)));

            m_layerEyeFov.RenderPose[eye] = eyePose;
        }
        glDisable(GL_SCISSOR_TEST);

        // Grab a copy of the left eye's undistorted render output for presentation
        // to the desktop window instead of the barrel distorted mirror texture.
        // This blit, while cheap, could cost some framerate to the HMD.
        // An over-the-shoulder view is another option, at a greater performance cost.
        if (m_mirror == MirrorUndistorted)
        {
            if (eye == ovrEyeType::ovrEye_Left)
            {
                BlitLeftEyeRenderToUndistortedMirrorTexture();
            }
        }

        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    // Submit layers to HMD for display
    std::vector<const ovrLayerHeader*> layers;
    layers.push_back(&m_layerEyeFov.Header);
    if (m_tweakbarQuad.m_showQuadInWorld)
    {
        _DrawToTweakbarQuad();

        // Display is a const function, but we cast that away to
        // be able to write quad's in-world pose each frame.
        ///@todo Update this in a separate non-const function -
        // it doesn't have to be as fresh as head pose does.
        ovrPosef& qpc = const_cast<ovrPosef&>(m_tweakbarQuad.m_layerQuad.QuadPoseCenter);
        const glm::vec3& qp = m_tweakbarQuad.m_quadLocation;
        qpc.Position = { qp.x, qp.y, qp.z };
        glm::quat qo = glm::quat();
        qo = glm::rotate(qo, m_tweakbarQuad.m_quadRotation.x, glm::vec3(1.f, 0.f, 0.f));
        qo = glm::rotate(qo, m_tweakbarQuad.m_quadRotation.y, glm::vec3(0.f, 1.f, 0.f));
        qpc.Orientation = { qo.x, qo.y, qo.z, qo.w };

        layers.push_back(&m_tweakbarQuad.m_layerQuad.Header);
    }

    if (m_secondQuad.m_showQuadInWorld)
    {
        _DrawToSecondQuad();
        ovrPosef& qpc = const_cast<ovrPosef&>(m_secondQuad.m_layerQuad.QuadPoseCenter);
        const glm::vec3& qp = m_secondQuad.m_quadLocation;
        qpc.Position = { qp.x, qp.y, qp.z };
        glm::quat qo = glm::quat();
        qo = glm::rotate(qo, m_secondQuad.m_quadRotation.x, glm::vec3(1.f, 0.f, 0.f));
        qo = glm::rotate(qo, m_secondQuad.m_quadRotation.y, glm::vec3(0.f, 1.f, 0.f));
        qpc.Orientation = { qo.x, qo.y, qo.z, qo.w };

        layers.push_back(&m_secondQuad.m_layerQuad.Header);
    }


    ovrViewScaleDesc viewScaleDesc;
    viewScaleDesc.HmdToEyeViewOffset[0] = m_eyeOffsets[0];
    viewScaleDesc.HmdToEyeViewOffset[1] = m_eyeOffsets[1];
    viewScaleDesc.HmdSpaceToWorldScaleInMeters = 1.f;
    const ovrResult result = ovr_SubmitFrame(hmd, m_frameIndex, &viewScaleDesc, &layers[0], layers.size());
    if (result == ovrSuccess_NotVisible)
    {
        LOG_INFO("ovr_SubmitFrame returned ovrSuccess_NotVisible");
        ///@todo Enter a lower-power, polling "no focus" mode
    }
    else if (result == ovrError_DisplayLost)
    {
        LOG_INFO("ovr_SubmitFrame returned ovrError_DisplayLost");
        ///@todo Tear down textures and session and re-create
    }
    ++m_frameIndex;

    // Increment counters in each swap texture set
    for (ovrEyeType eye = ovrEyeType::ovrEye_Left;
        eye < ovrEyeType::ovrEye_Count;
        eye = static_cast<ovrEyeType>(eye + 1))
    {
        ovrSwapTextureSet& swapSet = *m_pTexSet[eye];
        ++swapSet.CurrentIndex %= swapSet.TextureCount;
    }
    if (m_tweakbarQuad.m_showQuadInWorld)
    {
        ovrSwapTextureSet& swapSet = *m_tweakbarQuad.m_pQuadTex;
        ++swapSet.CurrentIndex %= swapSet.TextureCount;
    }
    if (m_secondQuad.m_showQuadInWorld)
    {
        ovrSwapTextureSet& swapSet = *m_secondQuad.m_pQuadTex;
        ++swapSet.CurrentIndex %= swapSet.TextureCount;
    }

    // Blit output to main app window to show something on screen in addition
    // to what's in the Rift. This could optionally be the distorted texture
    // from the OVR SDK's mirror texture, or perhaps a single eye's undistorted
    // view, or even a third-person render(at a performance cost).
    if (m_mirror != MirrorNone)
    {
        glViewport(0, 0, m_appWindowSize.w, m_appWindowSize.h);
        const FBO& srcFBO = m_mirror == MirrorDistorted ? m_mirrorFBO : m_undistortedFBO;
        glBindFramebuffer(GL_READ_FRAMEBUFFER, srcFBO.id);
        glBlitFramebuffer(
            0, srcFBO.h, srcFBO.w, 0,
            0, 0, m_appWindowSize.w, m_appWindowSize.h,
            GL_COLOR_BUFFER_BIT, GL_NEAREST);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
    }
    else
    {
        // Just a clear of the mirror window shouldn't be so expensive
        // as to hurt frame rate.
        glClearColor(0.f, 0.f, 0.f, 0.f);
        glClear(GL_COLOR_BUFFER_BIT);
    }
}
