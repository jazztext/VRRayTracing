// glfw_main.cpp
// With humongous thanks to cThrough 2014 (Daniel Dekkers)

#undef USE_ANTTWEAKBAR

#include <GL/glew.h>

#if defined(_WIN32)
#  include <Windows.h>
#  define GLFW_EXPOSE_NATIVE_WIN32
#  define GLFW_EXPOSE_NATIVE_WGL
#elif defined(__linux__)
#  include <X11/X.h>
#  include <X11/extensions/Xrandr.h>
#  define GLFW_EXPOSE_NATIVE_X11
#  define GLFW_EXPOSE_NATIVE_GLX
#endif

#include <GLFW/glfw3.h>

#if defined(_WIN32) || defined(_LINUX)
#  include <GLFW/glfw3native.h>
#endif

#include <glm/gtc/type_ptr.hpp>

#ifdef USE_ANTTWEAKBAR
#  include <AntTweakBar.h>
#endif

#include <stdio.h>
#include <string.h>
#include <sstream>
#include <algorithm>

#include <unistd.h>

#include "Scene/pathtracer.h"

#include <CMU462/CMU462.h>
#include <CMU462/viewer.h>

#include <application.h>


#if defined(USE_OSVR)
#include "OsvrAppSkeleton.h"
#elif defined(USE_OCULUSSDK)
#include "OVRSDK05AppSkeleton.h"
#include <OVR.h>
#include <GL/MatrixFunctions.h>
#else
#include "AppSkeleton.h"
#endif

#include "RenderingMode.h"
//#include "Timer.h"
//#include "FPSTimer.h"
#include "Logger.h"

#if defined(USE_OSVR)
OsvrAppSkeleton g_app;
#elif defined(USE_OCULUSSDK)
OVRSDK05AppSkeleton g_app;
#else
AppSkeleton g_app;
#endif

#ifndef PROJECT_NAME
// This macro should be defined in CMakeLists.txt
#define PROJECT_NAME "RiftSkeleton"
#endif

RenderingMode g_renderMode;
//Timer g_timer;
double g_lastFrameTime = 0.0;
//FPSTimer g_fps;
//Timer g_logDumpTimer;

int m_keyStates[GLFW_KEY_LAST];

// mouse motion internal state
int oldx, oldy, newx, newy;
int which_button = -1;
int modifier_mode = 0;

ShaderWithVariables g_auxPresent;
GLFWwindow* g_pHMDWindow = NULL;
GLFWwindow* g_AuxWindow = NULL;
int g_auxWindow_w = 1920 / 2;
int g_auxWindow_h = 587;

int g_joystickIdx = -1;

float g_fpsSmoothingFactor = 0.02f;
float g_fpsDeltaThreshold = 5.0f;
bool g_dynamicallyScaleFBO = false;
int g_targetFPS = 100;
bool g_drawToAuxWindow = false;
bool g_allowPitch = false;
bool g_allowRoll = false;

ovrVector3f initialPos[2];

#ifdef USE_ANTTWEAKBAR
TwBar* g_pTweakbar = NULL;
#endif

GLFWwindow* initializeAuxiliaryWindow(GLFWwindow* pRiftWindow);
void destroyAuxiliaryWindow(GLFWwindow* pAuxWindow);

// Set VSync is framework-dependent and has to come before the include
///@param state 0=off, 1=on, -1=adaptive
// Set vsync for both contexts.
static void SetVsync(int state)
{
    LOG_INFO("SetVsync(%d)", state);

    // Since AuxWindow holds the tweakbar, this should never be NULL
    if (g_AuxWindow != NULL)
    {
        glfwMakeContextCurrent(g_AuxWindow);
        glfwSwapInterval(state);
    }
    glfwMakeContextCurrent(g_pHMDWindow);
    glfwSwapInterval(state);
}

#include "main_include.cpp"

static void ErrorCallback(int p_Error, const char* p_Description)
{
    (void)p_Error;
    (void)p_Description;
    LOG_INFO("ERROR: %d, %s", p_Error, p_Description);
}


void keyboard(GLFWwindow* pWindow, int key, int codes, int action, int mods)
{
    (void)pWindow;
    (void)codes;

    if ((key > -1) && (key <= GLFW_KEY_LAST))
    {
        m_keyStates[key] = action;
        //printf("key ac  %d %d\n", key, action);
    }

    const float f = 0.9f;
    const float ff = 0.99f;

    if (action == GLFW_PRESS)
    {
    switch (key)
    {
        default:
            g_app.DismissHealthAndSafetyWarning();
            break;

        case GLFW_KEY_F1:
            if (mods & GLFW_MOD_CONTROL)
            {
                g_renderMode.toggleRenderingTypeReverse();
            }
            else
            {
                g_renderMode.toggleRenderingType();
            }
            LOG_INFO("Render Type: %d", g_renderMode.outputType);
            break;

        case GLFW_KEY_F2:
            g_renderMode.toggleRenderingTypeMono();
            LOG_INFO("Render Type: %d", g_renderMode.outputType);
            break;

        case GLFW_KEY_F3:
            g_renderMode.toggleRenderingTypeHMD();
            LOG_INFO("Render Type: %d", g_renderMode.outputType);
            break;

        case GLFW_KEY_F4:
            g_renderMode.toggleRenderingTypeDistortion();
            LOG_INFO("Render Type: %d", g_renderMode.outputType);
            break;

        case GLFW_KEY_F5: g_dynamicallyScaleFBO = false; g_app.SetFBOScale(f * g_app.GetFBOScale()); break;
        case GLFW_KEY_F6: g_dynamicallyScaleFBO = false; g_app.SetFBOScale(ff * g_app.GetFBOScale()); break;
        case GLFW_KEY_F7: g_dynamicallyScaleFBO = false; g_app.SetFBOScale((1.f/ff) * g_app.GetFBOScale()); break;
        case GLFW_KEY_F8: g_dynamicallyScaleFBO = false; g_app.SetFBOScale((1.f/f) * g_app.GetFBOScale()); break;

        case GLFW_KEY_F9: SetVsync(0); break;
        case GLFW_KEY_F10: SetVsync(1); break;
        case GLFW_KEY_F11: SetVsync(-1); break;

        case GLFW_KEY_DELETE: g_dynamicallyScaleFBO = !g_dynamicallyScaleFBO; break;

        case '`':
            ///@todo Is there a way to create an auxiliary window in Direct to rift mode?
            /// The call to glfwCreateWindow crashes the app in Win7.
            if (
                (g_app.UsingDirectMode() == false) ||
                (g_app.UsingDebugHmd() == true))
            {
                if (g_AuxWindow == NULL)
                {
                    LOG_INFO("Creating auxiliary window.");
                    g_AuxWindow = initializeAuxiliaryWindow(g_pHMDWindow);
                }
                else
                {
                    LOG_INFO("Destroying auxiliary window.");
                    destroyAuxiliaryWindow(g_AuxWindow);
                    glfwMakeContextCurrent(g_pHMDWindow);
                }
            }
            break;

        case GLFW_KEY_SPACE:
            g_app.RecenterPose();
            break;

        case 'R':
            g_app.ResetChassisTransformations();
            break;

        /* VRRT Code */
        case 'W':
              initialPos[0].z -= 0.1;
              initialPos[1].z -= 0.1;
//            g_app.m_scene.app->camera.move_forward(0.1);
            
//            g_app.m_scene.app->camera.rotate_by(0, 0.1);
            break;
        case 'S':
            initialPos[0].z += 0.1;
            initialPos[1].z += 0.1;
//           g_app.m_scene.app->camera.rotate_by(0, -0.1);
            break;

        case 'A':
            initialPos[0].x -= 0.1;
            initialPos[1].x -= 0.1;
//            g_app.m_scene.app->camera.move_by(-10, 0, 1);
            g_app.m_scene.app->camera.rotate_by(0.1, 0);
            break;
        case 'D':
            initialPos[0].x += 0.1;
            initialPos[1].x += 0.1;
//            g_app.m_scene.app->camera.move_by(10, 0, 1);
            g_app.m_scene.app->camera.rotate_by(-0.1,0);
            break;
        /* END VRRT Code */

#ifdef USE_OCULUSSDK
        case 'V': g_app.ToggleVignette(); break;
        case 'T': g_app.ToggleTimeWarp(); break;
        case 'O': g_app.ToggleOverdrive(); break;
        case 'L': g_app.ToggleLowPersistence(); break;
        case 'M': g_app.ToggleMirrorToWindow(); break;
        case 'P': g_app.ToggleDynamicPrediction(); break;
#endif

        case GLFW_KEY_ESCAPE:
            if (g_AuxWindow == NULL)
            {
                // Clear the frame before calling all the destructors - even a few
                // frames worth of frozen video is enough to cause discomfort!
                ///@note This does not seem to work in Direct mode.
                glClearColor(58.f/255.f, 110.f/255.f, 165.f/255.f, 1.f); // Win7 default desktop color
                glClear(GL_COLOR_BUFFER_BIT);
                glfwSwapBuffers(g_pHMDWindow);
                glClear(GL_COLOR_BUFFER_BIT);
                glfwSwapBuffers(g_pHMDWindow);

                g_app.exitVR();
                glfwDestroyWindow(g_pHMDWindow);
                glfwTerminate();
                exit(0);
            }
            else
            {
                destroyAuxiliaryWindow(g_AuxWindow);
                glfwMakeContextCurrent(g_pHMDWindow);
            }
            break;
        }
    }

    //g_app.keyboard(key, action, 0,0);

    const glm::vec3 forward(0.f, 0.f, -1.f);
    const glm::vec3 up(0.f, 1.f, 0.f);
    const glm::vec3 right(1.f, 0.f, 0.f);
    // Handle keyboard movement(WASD keys)
    glm::vec3 keyboardMove(0.0f, 0.0f, 0.0f);
    if (m_keyStates['W'] != GLFW_RELEASE) { keyboardMove += forward; }
    if (m_keyStates['S'] != GLFW_RELEASE) { keyboardMove -= forward; }
    if (m_keyStates['A'] != GLFW_RELEASE) { keyboardMove -= right; }
    if (m_keyStates['D'] != GLFW_RELEASE) { keyboardMove += right; }
    if (m_keyStates['Q'] != GLFW_RELEASE) { keyboardMove -= up; }
    if (m_keyStates['E'] != GLFW_RELEASE) { keyboardMove += up; }
    if (m_keyStates[GLFW_KEY_UP] != GLFW_RELEASE) { keyboardMove += forward; }
    if (m_keyStates[GLFW_KEY_DOWN] != GLFW_RELEASE) { keyboardMove -= forward; }
    if (m_keyStates[GLFW_KEY_LEFT] != GLFW_RELEASE) { keyboardMove -= right; }
    if (m_keyStates[GLFW_KEY_RIGHT] != GLFW_RELEASE) { keyboardMove += right; }

    float mag = 1.0f;
    if (m_keyStates[GLFW_KEY_LEFT_SHIFT] != GLFW_RELEASE) mag *= 0.1f;
    if (m_keyStates[GLFW_KEY_LEFT_CONTROL] != GLFW_RELEASE) mag *= 10.0f;
    if (m_keyStates[GLFW_KEY_RIGHT_SHIFT] != GLFW_RELEASE) mag *= 0.1f;
    if (m_keyStates[GLFW_KEY_RIGHT_CONTROL] != GLFW_RELEASE) mag *= 10.0f;

    // Yaw keys
    g_app.m_keyboardYaw = 0.0f;
    const float dyaw = 0.5f * mag; // radians at 60Hz timestep
    if (m_keyStates['1'] != GLFW_RELEASE) { g_app.m_keyboardYaw = -dyaw; }
    if (m_keyStates['3'] != GLFW_RELEASE) { g_app.m_keyboardYaw = dyaw; }

    // Pitch and roll controls - if yaw is VR poison,
    // this is torture and death!
    g_app.m_keyboardDeltaPitch = 0.0f;
    g_app.m_keyboardDeltaRoll = 0.0f;
    if (g_allowPitch)
    {
        if (m_keyStates['2'] != GLFW_RELEASE) { g_app.m_keyboardDeltaPitch = -dyaw; }
        if (m_keyStates['X'] != GLFW_RELEASE) { g_app.m_keyboardDeltaPitch = dyaw; }
    }
    if (g_allowRoll)
    {
        if (m_keyStates['Z'] != GLFW_RELEASE) { g_app.m_keyboardDeltaRoll = -dyaw; }
        if (m_keyStates['C'] != GLFW_RELEASE) { g_app.m_keyboardDeltaRoll = dyaw; }
    }

    g_app.m_keyboardMove = mag * keyboardMove;
}

void joystick()
{
    if (g_joystickIdx == -1)
        return;

    ///@todo Do these calls take time? We can move them out if so
    int joyStick1Present = GL_FALSE;
    joyStick1Present = glfwJoystickPresent(g_joystickIdx);
    if (joyStick1Present != GL_TRUE)
        return;

    // Poll joystick
    int numAxes = 0;
    const float* pAxisStates = glfwGetJoystickAxes(g_joystickIdx, &numAxes);
    if (numAxes < 2)
        return;

    int numButtons = 0;
    const unsigned char* pButtonStates = glfwGetJoystickButtons(g_joystickIdx, &numButtons);
    if (numButtons < 8)
        return;

    // Map joystick buttons to move directions
    const glm::vec3 moveDirsGravisGamepadPro[8] = {
        glm::vec3(-1.f,  0.f,  0.f),
        glm::vec3( 0.f,  0.f,  1.f),
        glm::vec3( 1.f,  0.f,  0.f),
        glm::vec3( 0.f,  0.f, -1.f),
        glm::vec3( 0.f,  1.f,  0.f),
        glm::vec3( 0.f,  1.f,  0.f),
        glm::vec3( 0.f, -1.f,  0.f),
        glm::vec3( 0.f, -1.f,  0.f),
    };

    // Xbox controller layout in glfw:
    // numAxes 5, numButtons 14
    // 0 A (down position)
    // 1 B (right position)
    // 2 X (left position)
    // 3 Y (up position)
    // 4 L bumper
    // 5 R bumper
    // 6 Back (left center)
    // 7 Start (right center)
    // Axis 0 1 Left stick x y
    // Axis 2 triggers, left positive right negative
    // Axis 3 4 right stick x y
    const glm::vec3 moveDirsXboxController[8] = {
        glm::vec3( 0.f,  0.f,  1.f),
        glm::vec3( 1.f,  0.f,  0.f),
        glm::vec3(-1.f,  0.f,  0.f),
        glm::vec3( 0.f,  0.f, -1.f),
        glm::vec3( 0.f, -1.f,  0.f),
        glm::vec3( 0.f,  1.f,  0.f),
        glm::vec3( 0.f,  0.f,  0.f),
        glm::vec3( 0.f,  0.f,  0.f),
    };

    ///@todo Different mappings for different controllers.
    const glm::vec3* moveDirs = moveDirsGravisGamepadPro;
    // Take an educated guess that this is an Xbox controller - glfw's
    // id string says "Microsoft PC Joystick" for most gamepad types.
    if (numAxes == 5 && numButtons == 14)
    {
        moveDirs = moveDirsXboxController;
    }

    glm::vec3 joystickMove(0.0f, 0.0f, 0.0f);
    for (int i=0; i<std::min(8,numButtons); ++i)
    {
        if (pButtonStates[i] == GLFW_PRESS)
        {
            joystickMove += moveDirs[i];
        }
    }

    float mag = 1.f;
    if (numAxes > 2)
    {
        mag = pow(10.f, pAxisStates[2]);
    }
    g_app.m_joystickMove = mag * joystickMove;

    float x_move = pAxisStates[0];
    const float deadzone = 0.2f;
    if (fabs(x_move) < deadzone)
        x_move = 0.0f;
    g_app.m_joystickYaw = 0.5f * static_cast<float>(x_move);
}

void mouseDown(GLFWwindow* pWindow, int button, int action, int mods)
{
    (void)mods;

    double xd, yd;
    glfwGetCursorPos(pWindow, &xd, &yd);
    const int x = static_cast<int>(xd);
    const int y = static_cast<int>(yd);

    which_button = button;
    oldx = newx = x;
    oldy = newy = y;
    if (action == GLFW_RELEASE)
    {
        which_button = -1;
    }
    g_app.OnMouseButton(button, action);
}

void mouseMove(GLFWwindow* pWindow, double xd, double yd)
{
    glfwGetCursorPos(pWindow, &xd, &yd);
    const int x = static_cast<int>(xd);
    const int y = static_cast<int>(yd);

    oldx = newx;
    oldy = newy;
    newx = x;
    newy = y;
    const int mmx = x-oldx;
    const int mmy = y-oldy;

    g_app.m_mouseDeltaYaw = 0.0f;
    g_app.m_mouseMove = glm::vec3(0.0f);

    if (which_button == GLFW_MOUSE_BUTTON_1)
    {
        const float spinMagnitude = 0.05f;
        g_app.m_mouseDeltaYaw += static_cast<float>(mmx) * spinMagnitude;
    }
    else if (which_button == GLFW_MOUSE_BUTTON_2) // Right click
    {
        const float moveMagnitude = 0.5f;
        g_app.m_mouseMove.x += static_cast<float>(mmx) * moveMagnitude;
        g_app.m_mouseMove.z += static_cast<float>(mmy) * moveMagnitude;
    }
    else if (which_button == GLFW_MOUSE_BUTTON_3) // Middle click
    {
        const float moveMagnitude = 0.5f;
        g_app.m_mouseMove.x += static_cast<float>(mmx) * moveMagnitude;
        g_app.m_mouseMove.y -= static_cast<float>(mmy) * moveMagnitude;
    }
    else
    {
        // Passive motion, no mouse button pressed
        g_app.OnMouseMove(static_cast<int>(x), static_cast<int>(y));
    }
}

void mouseWheel(GLFWwindow* pWindow, double x, double y)
{
    (void)pWindow;
    (void)x;

    const int delta = static_cast<int>(y);
    const float curscale = g_app.GetFBOScale();
    const float incr = 1.05f;
    g_app.SetFBOScale(curscale * pow(incr, static_cast<float>(delta)));
    if (fabs(x) > 0.)
    {
        g_app.OnMouseWheel(x,y);
    }
}

void resize(GLFWwindow* pWindow, int w, int h)
{
    (void)pWindow;
    g_app.resize(w,h);
}

void keyboard_Aux(GLFWwindow* pWindow, int key, int codes, int action, int mods)
{
#ifdef USE_ANTTWEAKBAR
    int ant = TwEventKeyGLFW(key, action);
    if (ant != 0)
        return;
#endif
    keyboard(pWindow, key, codes, action, mods);
}

void mouseDown_Aux(GLFWwindow* pWindow, int button, int action, int mods)
{
    (void)pWindow;
    (void)mods;

#ifdef USE_ANTTWEAKBAR
    int ant = TwEventMouseButtonGLFW(button, action);
    if (ant != 0)
        return;
#endif
    mouseDown(pWindow, button, action, mods);
}

void mouseMove_Aux(GLFWwindow* pWindow, double xd, double yd)
{
    (void)pWindow;

#ifdef USE_ANTTWEAKBAR
    int ant = TwEventMousePosGLFW(static_cast<int>(xd), static_cast<int>(yd));
    if (ant != 0)
        return;
#endif
    mouseMove(pWindow, xd, yd);
}

void mouseWheel_Aux(GLFWwindow* pWindow, double x, double y)
{
#ifdef USE_ANTTWEAKBAR
    static int scrollpos = 0;
    scrollpos += static_cast<int>(y);
    int ant = TwEventMouseWheelGLFW(scrollpos);
    if (ant != 0)
        return;
#endif
    mouseWheel(pWindow, x, y);
}

void resize_Aux(GLFWwindow* pWindow, int w, int h)
{
    (void)pWindow;
    g_auxWindow_w = w;
    g_auxWindow_h = h;

#ifdef USE_ANTTWEAKBAR
    ///@note This will break PaneScene's tweakbar positioning
    TwWindowSize(w, h);
#endif
}

void timestep()
{
//    const double absT = g_timer.seconds();
//    const double dt = absT - g_lastFrameTime;
//    g_lastFrameTime = absT;
    g_app.timestep(0,0);
}

void printGLContextInfo(GLFWwindow* pW)
{
    // Print some info about the OpenGL context...
    const int l_Major = glfwGetWindowAttrib(pW, GLFW_CONTEXT_VERSION_MAJOR);
    const int l_Minor = glfwGetWindowAttrib(pW, GLFW_CONTEXT_VERSION_MINOR);
    const int l_Profile = glfwGetWindowAttrib(pW, GLFW_OPENGL_PROFILE);
    if (l_Major >= 3) // Profiles introduced in OpenGL 3.0...
    {
        if (l_Profile == GLFW_OPENGL_COMPAT_PROFILE)
        {
            LOG_INFO("GLFW_OPENGL_COMPAT_PROFILE");
        }
        else
        {
            LOG_INFO("GLFW_OPENGL_CORE_PROFILE");
        }
    }
    (void)l_Minor;
    LOG_INFO("OpenGL: %d.%d", l_Major, l_Minor);
    LOG_INFO("Vendor: %s", reinterpret_cast<const char*>(glGetString(GL_VENDOR)));
    LOG_INFO("Renderer: %s", reinterpret_cast<const char*>(glGetString(GL_RENDERER)));
}

void initAuxPresentFboShader()
{
    g_auxPresent.initProgram("presentfbo");
    g_auxPresent.bindVAO();

    const float verts[] = {
        -1, -1,
        1, -1,
        1, 1,
        -1, 1
    };
    // The aspect ratio of one eye's view is half side-by-side(portrait), so we can chop
    // the top and bottom parts off to present something closer to landscape.
    const float texs[] = {
        0.0f, 0.25f,
        0.5f, 0.25f,
        0.5f, 0.75f,
        0.0f, 0.75f,
    };

    GLuint vertVbo = 0;
    glGenBuffers(1, &vertVbo);
    g_auxPresent.AddVbo("vPosition", vertVbo);
    glBindBuffer(GL_ARRAY_BUFFER, vertVbo);
    glBufferData(GL_ARRAY_BUFFER, 4*2*sizeof(GLfloat), verts, GL_STATIC_DRAW);
    glVertexAttribPointer(g_auxPresent.GetAttrLoc("vPosition"), 2, GL_FLOAT, GL_FALSE, 0, NULL);

    GLuint texVbo = 0;
    glGenBuffers(1, &texVbo);
    g_auxPresent.AddVbo("vTex", texVbo);
    glBindBuffer(GL_ARRAY_BUFFER, texVbo);
    glBufferData(GL_ARRAY_BUFFER, 4*2*sizeof(GLfloat), texs, GL_STATIC_DRAW);
    glVertexAttribPointer(g_auxPresent.GetAttrLoc("vTex"), 2, GL_FLOAT, GL_FALSE, 0, NULL);

    glEnableVertexAttribArray(g_auxPresent.GetAttrLoc("vPosition"));
    glEnableVertexAttribArray(g_auxPresent.GetAttrLoc("vTex"));

    glUseProgram(g_auxPresent.prog());
    {
        const glm::mat4 id(1.0f);
        glUniformMatrix4fv(g_auxPresent.GetUniLoc("mvmtx"), 1, false, glm::value_ptr(id));
        glUniformMatrix4fv(g_auxPresent.GetUniLoc("prmtx"), 1, false, glm::value_ptr(id));
    }
    glUseProgram(0);

    glBindVertexArray(0);
}

void presentSharedFboTexture()
{
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);

    glViewport(0, 0, g_auxWindow_w, g_auxWindow_h);

    // Present FBO to screen
    const GLuint prog = g_auxPresent.prog();
    glUseProgram(prog);
    g_auxPresent.bindVAO();
    {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, g_app.getRenderBufferTex());
        glUniform1i(g_auxPresent.GetUniLoc("fboTex"), 0);

        // This is the only uniform that changes per-frame
        const float fboScale = g_renderMode.outputType == RenderingMode::OVR_SDK ?
            1.0f :
            g_app.GetFBOScale();
        glUniform1f(g_auxPresent.GetUniLoc("fboScale"), fboScale);

        glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
    }
    glBindVertexArray(0);
    glUseProgram(0);
}

void displayToHMD()
{
    switch(g_renderMode.outputType)
    {
    case RenderingMode::Mono_Raw:
        g_app.display_raw();
        glfwSwapBuffers(g_pHMDWindow);
        break;

    case RenderingMode::Mono_Buffered:
        g_app.display_buffered();
        glfwSwapBuffers(g_pHMDWindow);
        break;

#if defined(USE_OSVR)
    case RenderingMode::SideBySide_Undistorted:
        g_app.display_stereo_undistorted();
        glfwSwapBuffers(g_pHMDWindow);
        break;

    case RenderingMode::OVR_SDK: ///@todo misnomer
    case RenderingMode::OVR_Client:
        g_app.display_stereo_distorted();
        glfwSwapBuffers(g_pHMDWindow);
        break;

#elif defined(USE_OCULUSSDK)
    case RenderingMode::SideBySide_Undistorted:
        g_app.display_stereo_undistorted();
        glfwSwapBuffers(g_pHMDWindow);
        break;

    case RenderingMode::OVR_SDK:
        g_app.display_sdk();
        // OVR SDK 05 will do its own swap
        break;

    case RenderingMode::OVR_Client:
        g_app.display_client();
        glfwSwapBuffers(g_pHMDWindow);
        break;
#endif //USE_OCULUSSDK

    default:
        LOG_ERROR("Unknown display type: %d", g_renderMode.outputType);
        break;
    }
}

///@return An auxiliary "control view" window to display a monoscopic view of the world
/// that the Rift user is inhabiting(on the primary VR window). Yes, this takes resources
/// away from the VR user's rendering and will lower the rendering throughput(MPx/sec)
/// available to the HMD. It should not negatively impact latency until frame rate drops
/// below the display's refresh rate(which will happen sooner with this extra load, but
/// can be tuned). Pixel fill can be tuned by adjusting the FBO render target size with
/// the mouse wheel, but vertex rate cannot and another render pass adds 50%.
///@todo A more palatable solution is to share the FBO render target between this and
/// the Rift window and just present the left half of it.
GLFWwindow* initializeAuxiliaryWindow(GLFWwindow* pRiftWindow)
{
    ///@todo Set size to half FBO target width
    GLFWwindow* pAuxWindow = glfwCreateWindow(g_auxWindow_w, g_auxWindow_h, "Control Window", NULL, pRiftWindow);
    if (pAuxWindow == NULL)
    {
        return NULL;
    }

    glfwMakeContextCurrent(pAuxWindow);
    {
        // Create context-specific data here
        initAuxPresentFboShader();
    }

    glfwSetMouseButtonCallback(pAuxWindow, mouseDown_Aux);
    glfwSetCursorPosCallback(pAuxWindow, mouseMove_Aux);
    glfwSetScrollCallback(pAuxWindow, mouseWheel_Aux);
    glfwSetKeyCallback(pAuxWindow, keyboard_Aux);
    glfwSetWindowSizeCallback(pAuxWindow, resize_Aux);

    // The window will be shown whether we do this or not (on Windows)...
    glfwShowWindow(pAuxWindow);

    glfwMakeContextCurrent(pRiftWindow);

    return pAuxWindow;
}

void destroyAuxiliaryWindow(GLFWwindow* pAuxWindow)
{
    glfwMakeContextCurrent(pAuxWindow);
    g_auxPresent.destroy();
    glfwDestroyWindow(pAuxWindow);
    g_AuxWindow = NULL;
}

// OpenGL debug callback
void GLAPIENTRY myCallback(
    GLenum source, GLenum type, GLuint id, GLenum severity,
    GLsizei length, const GLchar *msg,
    const void *data)
{
    switch (severity)
    {
    case GL_DEBUG_SEVERITY_HIGH:
    case GL_DEBUG_SEVERITY_MEDIUM:
    case GL_DEBUG_SEVERITY_LOW:
        LOG_INFO("[[GL Debug]] %x %x %x %x %s", source, type, id, severity, msg);
        break;
    case GL_DEBUG_SEVERITY_NOTIFICATION:
        break;
    }
}

int main(int argc, char** argv)
{
    printf("Welcome to VR RayTracer!\n");
#if defined(_WIN32)
    LOG_INFO("Windows build.");
#elif defined(_LINUX)
    LOG_INFO("Linux build.");
    LOG_INFO("DISPLAY=%s", getenv("DISPLAY"));
#elif defined(_MACOS)
    LOG_INFO("MacOS build.");
#endif


    bool useOpenGLCoreContext = false;
#ifdef USE_CORE_CONTEXT
    useOpenGLCoreContext = true;
#endif

    g_renderMode.outputType = RenderingMode::OVR_SDK;

    LOG_INFO("Using GLFW3 backend.");
    LOG_INFO("Compiled against GLFW %i.%i.%i",
        GLFW_VERSION_MAJOR,
        GLFW_VERSION_MINOR,
        GLFW_VERSION_REVISION);
    int major, minor, revision;
    glfwGetVersion(&major, &minor, &revision);
    LOG_INFO("Running against GLFW %i.%i.%i", major, minor, revision);
    LOG_INFO("glfwGetVersionString: %s", glfwGetVersionString());


    CMU462::AppConfig config; int opt;
    bool benchmark = false;
    bool renderToImage = false;
    bool runHarness = false;
    bool saveData = false;
    std::string filename = "Raytraced.png";
    char *fname;
    int numRays = 0;
    
    while ((opt = getopt(argc, argv, "s:l:t:m:e:h:b:ir:o:")) != -1) {
      switch (opt) {
        case 's':
          config.pathtracer_ns_aa = atoi(optarg);
          break;
        case 'l':
          config.pathtracer_ns_area_light = atoi(optarg);
          break;
        case 't':
          config.pathtracer_num_threads = atoi(optarg);
          break;
        case 'm':
          config.pathtracer_max_ray_depth = atoi(optarg);
          break;
        case 'b':
          benchmark = true;
          numRays = atoi(optarg);
          break;
        case 'i':
          renderToImage = true;
          break;
        case 'r': // Run on an input test harness
          runHarness = true;
          fname = optarg;
          break;
        case 'o': // Save the oculus data
          saveData = true;
          fname = optarg;
          break;
        default:
          usage(argv[0]);
          return 1;
     }
   }

   if (optind >= argc) {
     usage(argv[0]);
     return 1;
   }

   string sceneFilePath = argv[optind];
   printf("Input scene file: %s\n", sceneFilePath.c_str());

   //parse scene
   CMU462::Collada::SceneInfo *sceneInfo = new CMU462::Collada::SceneInfo();
   if (CMU462::Collada::ColladaParser::load(sceneFilePath.c_str(), sceneInfo) < 0) {
     delete sceneInfo;
     exit(0);
   }

   CMU462::Application *app = new CMU462::Application(config);
   app->load(sceneInfo);

   g_app.m_scene.app = app;

   delete sceneInfo;
 
   if (runHarness) {
     FILE *fp = fopen(fname, "r");
     char buf[1024];
     char outname[1024];
     int frame = 0;
     float f[3];
     float c[9];
     g_app.m_scene.initCuda();
     printf("Running harness now...\n");
     while (fgets(buf, 1023, fp)) {
       int toks = sscanf(buf, "%f %f %f %f %f %f %f %f %f %f %f %f", f,   f+1, f+2,
                                                                     c,   c+1, c+2,
                                                                     c+3, c+4, c+5,
                                                                     c+6, c+7, c+8);
/*       int toks = sscanf(buf, "%f %f %f %f %f %f %f %f %f %f %f %f\n", &app->camera.pos[0],    &app->camera.pos[1],    &app->camera.pos[2],
                                                          &app->camera.c2w[0][0], &app->camera.c2w[0][1], &app->camera.c2w[0][2],
                                                          &app->camera.c2w[1][0], &app->camera.c2w[1][1], &app->camera.c2w[1][2],
                                                          &app->camera.c2w[2][0], &app->camera.c2w[2][1], &app->camera.c2w[2][2]); */
       for (int i = 0; i < 3; i++) app->camera.pos[i] = f[i];
       for (int i = 0; i < 9; i++) app->camera.c2w[i / 3][i % 3] = c[i];
       sprintf(outname, "frames/rayTraced%04d.png", frame);
       g_app.m_scene.runStep(outname);
       printf("Finished frame %d!\n", frame);
       frame++;
     }
     fclose(fp);
     exit(0);
   }

   if (benchmark && renderToImage) {
     printf("Rendering to image takes precendence over benchmarking, running in -i mode.\n");
     benchmark = false;
   }

   // Various headless running modes.
   if (benchmark) {
     printf("Running benchmark test on %d rays...\n", numRays);
     app->benchmark(numRays);
     exit(0);
   }
   if (renderToImage) {
     printf("Rendering to image %s...\n", filename.c_str());
     app->pathtrace("RayTraced.png");
     exit(0);
   }

//   g_renderMode.outputType = RenderingMode::OVR_Client;


    /* RiftSkeleton CMD options, not necessary here.
    // Command line options
    for (int i=0; i<argc; ++i)
    {
        const std::string a = argv[i];
        LOG_INFO("argv[%d]: %s", i, a.c_str());
        if (!a.compare("-sdk"))
        {
            g_renderMode.outputType = RenderingMode::OVR_SDK;
        }
        else if (!a.compare("-client"))
        {
            g_renderMode.outputType = RenderingMode::OVR_Client;
        }
        else if (!a.compare("-core"))
        {
            useOpenGLCoreContext = true;
        }
        else if (!a.compare("-compat"))
        {
            useOpenGLCoreContext = false;
        }
    } */

#ifdef USE_OCULUSSDK
    g_app.initHMD();
#else
    g_renderMode.outputType = RenderingMode::Mono_Buffered;
#endif

    GLFWwindow* l_Window = NULL;
    glfwSetErrorCallback(ErrorCallback);
    if (!glfwInit())
    {
        exit(EXIT_FAILURE);
    }

    // Log system monitor information
    const GLFWmonitor* pPrimary = glfwGetPrimaryMonitor();
    int monitorCount = 0;
    GLFWmonitor** ppMonitors = glfwGetMonitors(&monitorCount);
    for (int i=0; i<monitorCount; ++i)
    {
        GLFWmonitor* pCur = ppMonitors[i];
        const GLFWvidmode* mode = glfwGetVideoMode(pCur);
        if (mode != NULL)
        {
            (void)pPrimary;
            LOG_INFO("Monitor #%d: %dx%d @ %dHz %s",
                i,
                mode->width,
                mode->height,
                mode->refreshRate,
                pCur==pPrimary ? "Primary":"");
        }
    }

    bool swapBackBufferDims = false;

    // Context setup - before window creation
    glfwWindowHint(GLFW_DEPTH_BITS, 16);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, useOpenGLCoreContext ? GLFW_OPENGL_CORE_PROFILE : GLFW_OPENGL_COMPAT_PROFILE);
    //glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#ifdef _DEBUG
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);
#endif

#if defined(USE_OSVR)
    LOG_INFO("USE_OSVR=1");
    std::string windowTitle = "";
    windowTitle = PROJECT_NAME "-GLFW-Osvr";

    if (g_app.UsingDebugHmd())
    {
        const hmdRes sz = { 800, 600 };
        // Create a normal, decorated application window
        LOG_INFO("Using Debug HMD mode.");
        windowTitle = PROJECT_NAME "-GLFW-DebugHMD";
        g_renderMode.outputType = RenderingMode::Mono_Buffered;

        l_Window = glfwCreateWindow(sz.w, sz.h, windowTitle.c_str(), NULL, NULL);
    }
    else
    {
        const hmdRes sz = {
            g_app.getHmdResolution().h,
            g_app.getHmdResolution().w
        };
        printf("size = (%d,%d)\n", g_app.getHmdResolution().w, g_app.getHmdResolution().h);
        exit(0);
        const winPos pos = g_app.getHmdWindowPos();
        g_renderMode.outputType = RenderingMode::SideBySide_Undistorted;

        LOG_INFO("Using Extended desktop mode.");
        windowTitle = PROJECT_NAME "-GLFW-Extended";

        LOG_INFO("Creating GLFW_DECORATED window %dx%d@%d,%d", sz.w, sz.h, pos.x, pos.y);
        glfwWindowHint(GLFW_DECORATED, 0);
        l_Window = glfwCreateWindow(sz.w, sz.h, windowTitle.c_str(), NULL, NULL);
        glfwWindowHint(GLFW_DECORATED, 1);
        glfwSetInputMode(l_Window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        glfwSetWindowPos(l_Window, pos.x, pos.y);
    }

#elif defined(USE_OCULUSSDK)
    LOG_INFO("USE_OCULUSSDK=1");
    ovrSizei sz = g_app.getHmdResolution();
    const ovrVector2i pos = g_app.getHmdWindowPos();
    std::string windowTitle = "";

    if (g_app.UsingDebugHmd() == true)
    {
        // Create a normal, decorated application window
        LOG_INFO("Using Debug HMD mode.");
        windowTitle = PROJECT_NAME "-GLFW-DebugHMD";
        g_renderMode.outputType = RenderingMode::Mono_Buffered;

        l_Window = glfwCreateWindow(sz.w, sz.h, windowTitle.c_str(), NULL, NULL);
    }
    else if (g_app.UsingDirectMode())
    {
        // HMD active - position undecorated window to fill HMD viewport
        LOG_INFO("Using Direct to Rift mode.");
        windowTitle = PROJECT_NAME "-GLFW-Direct";

        GLFWmonitor* monitor = glfwGetPrimaryMonitor();
        const GLFWvidmode* mode = glfwGetVideoMode(monitor);
        sz.w = mode->width;
        sz.h = mode->height;
        LOG_INFO("Creating window %dx%d@%d,%d", sz.w, sz.h, pos.x, pos.y);
        l_Window = glfwCreateWindow(sz.w, sz.h, windowTitle.c_str(), monitor, NULL);
        glfwSetInputMode(l_Window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

#ifdef _LINUX
        swapBackBufferDims = true;
#endif

#if defined(_WIN32)
        g_app.AttachToWindow((void*)glfwGetWin32Window(l_Window));
#endif
    }
    else
    {
        LOG_INFO("Using Extended desktop mode.");
        windowTitle = PROJECT_NAME "-GLFW-Extended";

        LOG_INFO("Creating GLFW_DECORATED window %dx%d@%d,%d", sz.w, sz.h, pos.x, pos.y);
        glfwWindowHint(GLFW_DECORATED, 0);
        l_Window = glfwCreateWindow(sz.w, sz.h, windowTitle.c_str(), NULL, NULL);
        glfwWindowHint(GLFW_DECORATED, 1);
        glfwSetInputMode(l_Window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        glfwSetWindowPos(l_Window, pos.x, pos.y);
    }

    resize(l_Window, sz.w, sz.h); // inform AppSkeleton of window size
#else
    const glm::vec2 sz(800, 600);
    // Create a normal, decorated application window
    LOG_INFO("Using No VR SDK.");
    const std::string windowTitle = PROJECT_NAME "-GLFW-NoVRSDK";
    g_renderMode.outputType = RenderingMode::Mono_Buffered;

    l_Window = glfwCreateWindow(sz.x, sz.y, windowTitle.c_str(), NULL, NULL);
#endif //USE_OSVR|USE_OCULUSSDK

    if (!l_Window)
    {
        LOG_INFO("Glfw failed to create a window. Exiting.");
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    // Required for SDK rendering (to do the buffer swap on its own)
#ifdef OVRSDK05
  #if defined(_WIN32)
    g_app.setWindow(glfwGetWin32Window(l_Window));
  #elif defined(__linux__)
    g_app.setWindow(NULL);//glfwGetX11Display());
  #endif
#endif //USE_OCULUSSDK

    glfwMakeContextCurrent(l_Window);
    glfwSetWindowSizeCallback(l_Window, resize);
    glfwSetMouseButtonCallback(l_Window, mouseDown);
    glfwSetCursorPosCallback(l_Window, mouseMove);
    glfwSetScrollCallback(l_Window, mouseWheel);
    glfwSetKeyCallback(l_Window, keyboard);

    memset(m_keyStates, 0, GLFW_KEY_LAST*sizeof(int));

    // joysticks
    for (int i = GLFW_JOYSTICK_1; i <= GLFW_JOYSTICK_LAST; ++i)
    {
        if (GL_FALSE == glfwJoystickPresent(i))
            continue;

        const char* pJoyName = glfwGetJoystickName(i);
        if (pJoyName == NULL)
            continue;

        int numAxes = 0;
        int numButtons = 0;
        glfwGetJoystickAxes(i, &numAxes);
        glfwGetJoystickButtons(i, &numButtons);

        LOG_INFO("Glfw opened Joystick #%d: %s w/ %d axes, %d buttons", i, pJoyName, numAxes, numButtons);
        if (g_joystickIdx == -1)
            g_joystickIdx = i;
    }

    printGLContextInfo(l_Window);
    glfwMakeContextCurrent(l_Window);
    g_pHMDWindow = l_Window;


    // Don't forget to initialize Glew, turn glewExperimental on to
    // avoid problems fetching function pointers...
    glewExperimental = GL_TRUE;
    const GLenum l_Result = glewInit();
    if (l_Result != GLEW_OK)
    {
        LOG_INFO("glewInit() error.");
        exit(EXIT_FAILURE);
    }

#ifdef _DEBUG
    // Debug callback initialization
    // Must be done *after* glew initialization.
    glDebugMessageCallback(myCallback, NULL);
    glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, NULL, GL_TRUE);
    glDebugMessageInsert(GL_DEBUG_SOURCE_APPLICATION, GL_DEBUG_TYPE_MARKER, 0,
        GL_DEBUG_SEVERITY_NOTIFICATION, -1 , "Start debugging");
    glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
#endif

#ifdef USE_ANTTWEAKBAR
    LOG_INFO("Using AntTweakbar.");
    TwInit(useOpenGLCoreContext ? TW_OPENGL_CORE : TW_OPENGL, NULL);
    InitializeBar();
#endif


    LOG_INFO("Calling initGL...");
    g_app.initGL();
    LOG_INFO("Calling initVR...");
    g_app.initVR(swapBackBufferDims);
    LOG_INFO("initVR(%d) complete.", swapBackBufferDims);

    LOG_INFO("Adding VRRT application info into OVR...");
    g_app.m_scene.app = app;

    LOG_INFO("Initializing CUDA...");
    g_app.m_scene.initCuda();

    std::vector<CMU462::StaticScene::Primitive*> primitives;
    for (CMU462::StaticScene::SceneObject *obj : app->scene->objects) {
      const std::vector<CMU462::StaticScene::Primitive*> &obj_prims = obj->get_primitives();
      primitives.reserve(primitives.size() + obj_prims.size());
      primitives.insert(primitives.end(), obj_prims.begin(), obj_prims.end());
    }
    CMU462::StaticScene::BVHAccel *bvh = new CMU462::StaticScene::BVHAccel(primitives, 4);
    g_app.m_scene.bvh = VRRT::BVHGPU(bvh);


    LOG_INFO("glfwWindowShouldClose: %d", glfwWindowShouldClose(l_Window));

    initialPos[0] = OVR::Vector3f(g_app.m_EyeRenderDesc[0].HmdToEyeViewOffset);
    initialPos[1] = OVR::Vector3f(g_app.m_EyeRenderDesc[1].HmdToEyeViewOffset);

    ovrTrackingState outHmdTrackingState;
    ovrPosef eyes[2];
    ovrHmd_GetEyePoses(g_app.m_Hmd, 0, initialPos, eyes, &outHmdTrackingState);

    initialPos[0] = eyes[0].Position;
    initialPos[1] = eyes[1].Position;

    initialPos[0].x = g_app.m_scene.app->camera.pos[0] - initialPos[0].x;
    initialPos[0].y = g_app.m_scene.app->camera.pos[1] - initialPos[0].y;
    initialPos[0].z = g_app.m_scene.app->camera.pos[2] - initialPos[0].z;
    initialPos[1].x = g_app.m_scene.app->camera.pos[0] - initialPos[1].x;
    initialPos[1].y = g_app.m_scene.app->camera.pos[1] - initialPos[1].y;
    initialPos[1].z = g_app.m_scene.app->camera.pos[2] - initialPos[1].z;

    g_app.DismissHealthAndSafetyWarning();

    FILE *fp = NULL;
    if (saveData) {
      fp = fopen(fname, "w");
    }

    while (!glfwWindowShouldClose(l_Window))
    {
        g_app.CheckForTapToDismissHealthAndSafetyWarning();
        glfwPollEvents();

        ovrVector3f e2v[2] = {
            OVR::Vector3f(g_app.m_EyeRenderDesc[0].HmdToEyeViewOffset),
            OVR::Vector3f(g_app.m_EyeRenderDesc[1].HmdToEyeViewOffset), };

//        g_app.m_scene.outFbo = 0;
        g_app.m_scene.outFbo = g_app.m_renderBuffer.id;
//        ovrHmd_BeginFrame(g_app.m_Hmd, 0);
        ovrHmd_GetEyePoses(g_app.m_Hmd, 0, e2v, eyes, &outHmdTrackingState);

        /* Get the rotation data (should be the same for each eye) */
        glm::mat4 viewLocal = makeMatrixFromPose(eyes[0]);
        for (int i = 0; i < 3; i++) {
          for (int j = 0; j < 3; j++) g_app.m_scene.app->camera.c2w[i][j] = viewLocal[i][j];
        }


        for (int i = 0; i < 2; i++) {
          g_app.m_scene.app->camera.pos[0] = eyes[i].Position.x + initialPos[i].x;
          g_app.m_scene.app->camera.pos[1] = eyes[i].Position.y + initialPos[i].y;
          g_app.m_scene.app->camera.pos[2] = eyes[i].Position.z + initialPos[i].z;

          g_app.m_scene.eye = i;
          timestep();
          g_app.m_scene.RenderForOneEye(NULL,NULL);
        }

        if (saveData) {
          fprintf(fp, "%f %f %f ", g_app.m_scene.app->camera.pos[0], 
                                   g_app.m_scene.app->camera.pos[1],
                                   g_app.m_scene.app->camera.pos[2]);
          for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
              fprintf(fp, "%f ", g_app.m_scene.app->camera.c2w[i][j]);
            }
          }
          fprintf(fp, "\n");
        }


//        ovrHmd_EndFrame(g_app.m_Hmd, eyes, (const ovrTexture*)fbos);
        displayToHMD();
    }

    g_app.exitVR();
    glfwDestroyWindow(l_Window);
    glfwTerminate();

    exit(EXIT_SUCCESS);
}
