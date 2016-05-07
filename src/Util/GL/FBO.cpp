// FBO.cpp

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

#include <GL/glew.h>
#include "FBO.h"

// FrameBuffer Object
// Allows rendering to texture.
void allocateFBO(FBO& f, int w, int h)
{
    // Delete old textures if they exist
    deallocateFBO(f);

    f.w = w;
    f.h = h;

    glGenFramebuffers(1, &f.id);
    glBindFramebuffer(GL_FRAMEBUFFER, f.id);
    
#if 0
    // Depth buffer render target
    glGenRenderbuffers(1, &f.depth);
    glBindRenderbuffer(GL_RENDERBUFFER, f.depth);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, w, h);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER,
                              GL_DEPTH_ATTACHMENT,
                              GL_RENDERBUFFER,
                              f.depth);

    // Depth buffer texture target
    glGenTextures(1, &f.depth);
    glBindTexture(GL_TEXTURE_2D, f.depth);
    {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        //glTexParameteri( GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE, GL_INTENSITY); //deprecated, out in 3.1
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT,
                     w, h, 0,
                     GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, NULL);
    }
    glBindTexture(GL_TEXTURE_2D, 0);

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, f.depth, 0);
#endif


    // Texture render target
    glGenTextures(1, &f.tex);
    glBindTexture(GL_TEXTURE_2D, f.tex);
    {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8,
                     w, h, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    }
    glBindTexture(GL_TEXTURE_2D, 0);

    glFramebufferTexture2D(GL_FRAMEBUFFER,
                           GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D,
                           f.tex, 0);

    // Check status
    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE)
    {
        //printf("BufferStructs.cpp: Framebuffer is incomplete with status %d\n", status);
        //assert(false);
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void deallocateFBO(FBO& f)
{
    glDeleteFramebuffers(1, &f.id), f.id = 0;
    glDeleteTextures(1, &f.tex), f.tex = 0;
#if 0
    glDeleteRenderbuffers(1, &f.depth), f.depth = 0;
#else
    glDeleteTextures(1, &f.depth), f.depth = 0;
#endif
}

///@note This hack is to get around lack of glPushAttrib
static GLint s_vp[4];

// Set viewport here, then restore it in unbind
void bindFBO(const FBO& f, float fboScale)
{
    glBindFramebuffer(GL_FRAMEBUFFER, f.id);
    glGetIntegerv(GL_VIEWPORT, &s_vp[0]);

    // Add 1 to the viewport sizes here to mitigate the edge effects on the render buffer -
    // this way we render all the way out to the borders rather than leaving an unsightly gap.
    glViewport(
        0, 0,
        static_cast<int>(f.w * fboScale) + 1,
        static_cast<int>(f.h * fboScale) + 1);
}

void unbindFBO()
{
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    ///@warning We hope here that the FBO being unbound is the last one that was bound.
    glViewport(s_vp[0], s_vp[1], s_vp[2], s_vp[3]);
}
