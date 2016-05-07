/* GENERATED FILE - DO NOT EDIT!
 * Created by hardcode_shaders.py.
 *
 */

#include <map>

const char* basic_frag = 
    "// basic.frag\n"
    "#version 330\n"
    "in vec3 vfColor;\n"
    "out vec4 fragColor;\n"
    "void main()\n"
    "{\n"
    "    fragColor = vec4(vfColor, 1.0);\n"
    "}\n"
;

const char* basic_vert = 
    "// basic.vert\n"
    "#version 330\n"
    "in vec4 vPosition;\n"
    "in vec4 vColor;\n"
    "out vec3 vfColor;\n"
    "uniform mat4 mvmtx;\n"
    "uniform mat4 prmtx;\n"
    "void main()\n"
    "{\n"
    "    vfColor = vColor.xyz;\n"
    "    gl_Position = prmtx * mvmtx * vPosition;\n"
    "}\n"
;

const char* basicplane_frag = 
    "// basicplane.frag\n"
    "// Apply a simple black and white checkerboard pattern to a quad\n"
    "// with texture coordinates in the unit interval.\n"
    "#version 330\n"
    "in vec2 vfTexCoord;\n"
    "out vec4 fragColor;\n"
    "float pi = 3.14159265358979323846;\n"
    "void main()\n"
    "{\n"
    "    float freq = 16.0 * pi;\n"
    "    float lum = floor(1.0 + 2.0*sin(freq*vfTexCoord.x) * sin(freq*vfTexCoord.y));\n"
    "    fragColor = vec4(vec3(lum), 1.0);\n"
    "}\n"
;

const char* basicplane_vert = 
    "// basicplane.vert\n"
    "#version 330\n"
    "in vec3 vPosition;\n"
    "in vec2 vTexCoord;\n"
    "out vec2 vfTexCoord;\n"
    "uniform mat4 mvmtx;\n"
    "uniform mat4 prmtx;\n"
    "void main()\n"
    "{\n"
    "    vfTexCoord = vTexCoord;\n"
    "    gl_Position = prmtx * mvmtx * vec4(vPosition, 1.0);\n"
    "}\n"
;

const char* hydrabase_frag = 
    "// hydrabase.frag\n"
    "#version 330\n"
    "in vec3 vfColor;\n"
    "out vec4 fragColor;\n"
    "void main()\n"
    "{\n"
    "    vec2 tc = vfColor.xy;\n"
    "    vec3 bc = vec3(tc, 1.0-tc.x-tc.y); // something like barycentric...\n"
    "    float mincomp = 2.0 * min(min(bc.x, bc.y), bc.z);\n"
    "    float mid = 0.4;\n"
    "    float halfthick = 0.05;\n"
    "    float band = smoothstep(mid-halfthick, mid, mincomp) * (1.0-smoothstep(mid, mid+halfthick, mincomp));\n"
    "    vec3 col = vec3(0.0, 1.0, 0.0);\n"
    "    fragColor = vec4(band*col, 1.0);\n"
    "}\n"
;

const char* hydrabase_vert = 
    "// hydrabase.vert\n"
    "#version 330\n"
    "in vec4 vPosition;\n"
    "in vec4 vColor;\n"
    "out vec3 vfColor;\n"
    "uniform mat4 mvmtx;\n"
    "uniform mat4 prmtx;\n"
    "void main()\n"
    "{\n"
    "    float radius = 0.0254;\n"
    "    vfColor = vColor.xyz;\n"
    "    gl_Position = prmtx * mvmtx * vec4(radius * normalize(vPosition.xyz), 1.0);\n"
    "}\n"
;

const char* presentfbo_frag = 
    "// presentfbo.frag\n"
    "#version 330\n"
    "in vec2 vfTex;\n"
    "out vec4 fragColor;\n"
    "uniform float fboScale;\n"
    "uniform sampler2D fboTex;\n"
    "void main()\n"
    "{\n"
    "    fragColor = texture(fboTex, vfTex * fboScale);\n"
    "}\n"
;

const char* presentfbo_vert = 
    "// presentfbo.vert\n"
    "#version 330\n"
    "in vec2 vPosition;\n"
    "in vec2 vTex;\n"
    "out vec2 vfTex;\n"
    "uniform mat4 mvmtx;\n"
    "uniform mat4 prmtx;\n"
    "void main()\n"
    "{\n"
    "    vfTex = vTex;\n"
    "    gl_Position = prmtx * mvmtx * vec4(vPosition, 0.0, 1.0);\n"
    "}\n"
;

const char* presentmesh_frag = 
    "// presentmesh.frag\n"
    "#version 330\n"
    "uniform float fboScale;\n"
    "uniform sampler2D fboTex;\n"
    "in vec2 vfTexR;\n"
    "in vec2 vfTexG;\n"
    "in vec2 vfTexB;\n"
    "in float vfColor;\n"
    "out vec4 fragColor;\n"
    "vec2 scaleAndFlip(vec2 tc)\n"
    "{\n"
    "    return fboScale * vec2(tc.x, 1.0-tc.y);\n"
    "}\n"
    "void main()\n"
    "{\n"
    "    float resR = texture(fboTex, scaleAndFlip(vfTexR)).r;\n"
    "    float resG = texture(fboTex, scaleAndFlip(vfTexG)).g;\n"
    "    float resB = texture(fboTex, scaleAndFlip(vfTexB)).b;\n"
    "    fragColor = vec4(vfColor * vec3(resR, resG, resB), 1.0);\n"
    "}\n"
;

const char* presentmesh_vert = 
    "// presentmesh.vert\n"
    "#version 330\n"
    "uniform vec2 EyeToSourceUVScale;\n"
    "uniform vec2 EyeToSourceUVOffset;\n"
    "uniform mat4 EyeRotationStart;\n"
    "uniform mat4 EyeRotationEnd;\n"
    "in vec4 vPosition;\n"
    "in vec2 vTexR;\n"
    "in vec2 vTexG;\n"
    "in vec2 vTexB;\n"
    "out vec2 vfTexR;\n"
    "out vec2 vfTexG;\n"
    "out vec2 vfTexB;\n"
    "out float vfColor;\n"
    "vec2 TimewarpTexCoord(vec2 TexCoord, mat4 rotMat)\n"
    "{\n"
    "    // Vertex inputs are in TanEyeAngle space for the R,G,B channels (i.e. after chromatic \n"
    "    // aberration and distortion). These are now \"real world\" vectors in direction (x,y,1) \n"
    "    // relative to the eye of the HMD. Apply the 3x3 timewarp rotation to these vectors.\n"
    "    vec3 transformed = (rotMat * vec4(TexCoord.xy, 1., 1.)).xyz;\n"
    "    // Project them back onto the Z=1 plane of the rendered images.\n"
    "    vec2 flattened = (transformed.xy / transformed.z);\n"
    "    // Scale them into ([0,0.5],[0,1]) or ([0.5,0],[0,1]) UV lookup space (depending on eye)\n"
    "    return EyeToSourceUVScale * flattened + EyeToSourceUVOffset;\n"
    "}\n"
    "void main()\n"
    "{\n"
    "    float timewarpLerpFactor = vPosition.z;\n"
    "    //mat4 lerpedEyeRot = mix(EyeRotationStart, EyeRotationEnd, timewarpLerpFactor);\n"
    "    mat4 lerpedEyeRot = (1.-timewarpLerpFactor)*EyeRotationStart + timewarpLerpFactor*EyeRotationEnd ;\n"
    "    vfTexR = TimewarpTexCoord(vTexR, lerpedEyeRot);\n"
    "    vfTexG = TimewarpTexCoord(vTexG, lerpedEyeRot);\n"
    "    vfTexB = TimewarpTexCoord(vTexB, lerpedEyeRot);\n"
    "    vfColor = vPosition.w;\n"
    "    gl_Position = vec4(vPosition.xy, 0.5, 1.0);\n"
    "}\n"
;

const char* ucolor_frag = 
    "// ucolor.frag\n"
    "#version 330\n"
    "uniform vec4 u_Color;\n"
    "out vec4 fragColor;\n"
    "void main()\n"
    "{\n"
    "    fragColor = u_Color;\n"
    "}\n"
;

const char* ucolor_vert = 
    "// ucolor.vert\n"
    "#version 330\n"
    "in vec4 vPosition;\n"
    "uniform mat4 mvmtx;\n"
    "uniform mat4 prmtx;\n"
    "void main()\n"
    "{\n"
    "    gl_Position = prmtx * mvmtx * vPosition;\n"
    "}\n"
;


std::map<std::string, std::string> g_shaderMap;


void initShaderList() {
    g_shaderMap["basic.frag"] = basic_frag;
    g_shaderMap["basic.vert"] = basic_vert;
    g_shaderMap["basicplane.frag"] = basicplane_frag;
    g_shaderMap["basicplane.vert"] = basicplane_vert;
    g_shaderMap["hydrabase.frag"] = hydrabase_frag;
    g_shaderMap["hydrabase.vert"] = hydrabase_vert;
    g_shaderMap["presentfbo.frag"] = presentfbo_frag;
    g_shaderMap["presentfbo.vert"] = presentfbo_vert;
    g_shaderMap["presentmesh.frag"] = presentmesh_frag;
    g_shaderMap["presentmesh.vert"] = presentmesh_vert;
    g_shaderMap["ucolor.frag"] = ucolor_frag;
    g_shaderMap["ucolor.vert"] = ucolor_vert;
}
