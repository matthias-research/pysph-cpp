#include "FluidRenderer.h"
#ifdef _WIN32
#include <Windows.h>
#endif
#include <iostream>
#include <cstring>
#include <cmath>

// CUDA kernel declaration
extern void launchCurvatureFlow(cudaSurfaceObject_t depthIn, cudaSurfaceObject_t depthOut,
                                int width, int height, float dt, float zContrib,
                                float projW, float projH);

//-----------------------------------------------------------------------------
// Shader sources
//-----------------------------------------------------------------------------

// Vertex shader for point sprites (shared by ball, depth, thickness)
static const char* pointSpriteVertexShader = R"(
#version 330 core
layout (location = 0) in vec4 aPos;

uniform mat4 modelViewProj;
uniform mat4 modelView;
uniform mat4 proj;
uniform float pointScale;
uniform float radius;

out vec3 eyePos;

void main() {
    vec4 eyeSpacePos = modelView * vec4(aPos.xyz, 1.0);
    gl_Position = proj * eyeSpacePos;
    eyePos = eyeSpacePos.xyz;
    
    float dist = length(eyeSpacePos.xyz);
    gl_PointSize = radius * (pointScale / dist);
}
)";

// Ball fragment shader - lit spheres (matching Python's ballFragment)
static const char* ballFragmentShader = R"(
#version 330 core
uniform mat4 proj;
uniform float radius;

in vec3 eyePos;
out vec4 fragColor;

void main() {
    vec2 coord = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(coord, coord);
    if (r2 > 1.0) discard;
    
    float h = sqrt(1.0 - r2);
    vec3 normal = normalize(vec3(coord.x, coord.y, h));
    
    // Simple directional light
    vec3 lightDir = normalize(vec3(0.577, 0.577, 0.577));
    float diffuse = max(0.0, dot(lightDir, normal));
    
    // Color by position (matching Python)
    vec3 baseColor = (eyePos / 10.0) * 0.5 + 0.5;
    
    fragColor = vec4(baseColor * diffuse, 1.0);
    
    // Update depth buffer for sphere surface
    vec4 clipPos = proj * vec4(eyePos + normal * radius, 1.0);
    gl_FragDepth = (clipPos.z / clipPos.w) * 0.5 + 0.5;
}
)";

// Depth fragment shader - outputs eye-space Z (matching Python's depthFragment)
static const char* depthFragmentShader = R"(
#version 330 core
uniform mat4 proj;
uniform float radius;
uniform float nearClip;
uniform float farClip;

in vec3 eyePos;
out float fragDepth;

float projectZ(float z) {
    return farClip * (z + nearClip) / (z * (farClip - nearClip));
}

void main() {
    vec2 coord = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(coord, coord);
    if (r2 > 1.0) discard;
    
    float h = sqrt(1.0 - r2);
    float z = eyePos.z + h * radius;  // Sphere surface Z (right-handed)
    
    fragDepth = z;  // Output unprojected Z for smoothing
    gl_FragDepth = projectZ(z) * 0.5 + 0.5;
}
)";

// Thickness fragment shader (matching Python's thicknessFragment)
static const char* thicknessFragmentShader = R"(
#version 330 core
out float fragThickness;

void main() {
    vec2 coord = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(coord, coord);
    if (r2 > 1.0) discard;
    
    fragThickness = 1.0;  // Additive blending accumulates thickness
}
)";

// Fullscreen quad vertex shader
static const char* fullscreenVertexShader = R"(
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexCoord;

out vec2 texCoord;

void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    texCoord = aTexCoord;
}
)";

// Blur fragment shader (matching Python's blurFragment)
static const char* blurFragmentShader = R"(
#version 330 core
uniform sampler2D tex;
uniform vec2 direction;
uniform float texelSize;

const float sigma = 5.0;
const int radius = 20;

in vec2 texCoord;
out float fragColor;

void main() {
    float sum = 0.0;
    float wsum = 0.0;
    
    for (int r = -radius; r <= radius; r++) {
        float sample = texture(tex, texCoord + float(r) * direction * texelSize).r;
        float v = float(r) / sigma;
        float w = exp(-v * v / 2.0);
        sum += sample * w;
        wsum += w;
    }
    
    fragColor = (wsum > 0.0) ? sum / wsum : 0.0;
}
)";

// Debug visualization shader - shows raw texture values
static const char* debugFragmentShader = R"(
#version 330 core
uniform sampler2D tex;
uniform int vizMode;  // 0=depth, 1=thickness, 2=normals

in vec2 texCoord;
out vec4 fragColor;

void main() {
    float val = texture(tex, texCoord).r;
    
    if (vizMode == 0) {
        // Depth: negative eye-space Z, map to visible range
        // Typical values are -5 to -30 (in front of camera)
        if (val == 0.0) {
            fragColor = vec4(0.0, 0.0, 0.0, 1.0);  // Black = no depth
        } else {
            float normalized = clamp(-val / 30.0, 0.0, 1.0);  // Map -30..0 to 0..1
            fragColor = vec4(normalized, normalized, normalized, 1.0);
        }
    } else if (vizMode == 1) {
        // Thickness: typically 0 to 10+
        float normalized = clamp(val / 10.0, 0.0, 1.0);
        fragColor = vec4(normalized, normalized * 0.5, 0.0, 1.0);  // Orange gradient
    } else {
        // Raw value
        fragColor = vec4(val, val, val, 1.0);
    }
}
)";

// Final compositing shader (matching Python's finalFragment)
static const char* finalFragmentShader = R"(
#version 330 core
uniform sampler2D depthTex;
uniform sampler2D thicknessTex;
uniform mat4 proj;
uniform mat4 modelView;
uniform float nearClip;
uniform float farClip;
uniform float radius;
uniform float projW;
uniform float projH;
uniform vec2 windowSize;

in vec2 texCoord;
out vec4 fragColor;

float projectZ(float z) {
    return farClip * (z + nearClip) / (z * (farClip - nearClip));
}

vec3 uvToEye(vec2 uv, float z) {
    uv = uv * 2.0 - 1.0;
    return vec3(-uv.x / projW, -uv.y / projH, 1.0) * z;
}

float diffZ(vec2 uv, vec2 offset) {
    float dp = texture(depthTex, uv + offset).r;
    float dm = texture(depthTex, uv - offset).r;
    if (dm == 0.0) return dp - texture(depthTex, uv).r;
    if (dp == 0.0) return 0.0;
    return (dp - dm) * 0.5;
}

vec3 computeNormal(vec2 uv) {
    float z = texture(depthTex, uv).r;
    vec2 texelSize = 1.0 / windowSize;
    
    float z_x = diffZ(uv, vec2(texelSize.x, 0.0));
    float z_y = diffZ(uv, vec2(0.0, texelSize.y));
    
    float Cx = -2.0 / (windowSize.x * projW);
    float Cy = -2.0 / (windowSize.y * projH);
    
    vec2 screenPos = uv * windowSize;
    float Wx = (windowSize.x - 2.0 * screenPos.x) / (windowSize.x * projW);
    float Wy = (windowSize.y - 2.0 * screenPos.y) / (windowSize.y * projH);
    
    vec3 dx = vec3(Cx * z + Wx * z_x, Wy * z_x, z_x);
    vec3 dy = vec3(Wx * z_y, Cy * z + Wy * z_y, z_y);
    
    vec3 normal = cross(dx, dy);
    float len = length(normal);
    
    // When viewing flat surface head-on, dx and dy become nearly parallel,
    // making the cross product unstable. Fall back to view-facing normal.
    if (len < 1e-6) {
        return vec3(0.0, 0.0, 1.0);  // Default: facing camera
    }
    normal = normal / len;
    
    // Transform to world space
    mat3 invModelView = transpose(mat3(modelView));
    return invModelView * normal;
}

void main() {
    float depth = texture(depthTex, texCoord).r;
    if (depth == 0.0) discard;
    
    gl_FragDepth = projectZ(depth) * 0.5 + 0.5;
    
    float thickness = texture(thicknessTex, texCoord).r * radius;
    vec3 normal = computeNormal(texCoord);
    
    // Beer's law absorption (water color)
    vec3 absorption = vec3(
        exp(-0.6 * thickness),   // Red absorbed most
        exp(-0.2 * thickness),   // Some green
        exp(-0.05 * thickness)   // Blue passes through
    );
    float alpha = 1.0 - exp(-3.0 * thickness);
    
    // Diffuse lighting
    vec3 lightDir = normalize(vec3(0.577, -0.577, 0.577));
    float diffuse = abs(dot(lightDir, normal)) * 0.5 + 0.5;
    
    // Specular (Schlick's approximation)
    vec3 pos3D = uvToEye(texCoord, depth);
    mat3 invModelView = transpose(mat3(modelView));
    pos3D = invModelView * pos3D;
    
    vec3 eyePos = -vec3(modelView[3]) * invModelView;
    vec3 viewDir = normalize(eyePos - pos3D);
    
    float normalReflectance = pow(clamp(dot(normal, lightDir), 0.0, 1.0), 6.0);
    float fresnel = normalReflectance + (1.0 - normalReflectance) * pow(1.0 - abs(dot(normal, viewDir)), 8.0);
    float specular = clamp(0.1 * thickness, 0.0, 1.0) * fresnel;
    
    fragColor = clamp(vec4(diffuse * absorption + specular, alpha), 0.0, 1.0);
}
)";

//-----------------------------------------------------------------------------
// Helper functions
//-----------------------------------------------------------------------------

static GLuint compileShader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char log[512];
        glGetShaderInfoLog(shader, 512, nullptr, log);
        std::cerr << "Shader compilation error: " << log << std::endl;
    }
    return shader;
}

static GLuint createProgram(const char* vertSrc, const char* fragSrc) {
    GLuint vert = compileShader(GL_VERTEX_SHADER, vertSrc);
    GLuint frag = compileShader(GL_FRAGMENT_SHADER, fragSrc);
    
    GLuint program = glCreateProgram();
    glAttachShader(program, vert);
    glAttachShader(program, frag);
    glLinkProgram(program);
    
    GLint success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char log[512];
        glGetProgramInfoLog(program, 512, nullptr, log);
        std::cerr << "Shader link error: " << log << std::endl;
    }
    
    glDeleteShader(vert);
    glDeleteShader(frag);
    return program;
}

static void checkGLError(const char* location) {
    GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
        std::cerr << "OpenGL error at " << location << ": 0x" << std::hex << err << std::dec << std::endl;
    }
}

//-----------------------------------------------------------------------------
// FluidRenderer implementation
//-----------------------------------------------------------------------------

FluidRenderer::FluidRenderer()
    : windowWidth(0)
    , windowHeight(0)
    , renderMode(RENDERMODE_BALLS)
    , smoothDepth(true)
    , blurThickness(true)
    , smoothingIterations(50)
    , smoothingDt(0.005f)
    , smoothingZContrib(10.0f)
    , debugMode(DEBUG_NONE)
    , shaderBall(0)
    , shaderDepth(0)
    , shaderThickness(0)
    , shaderBlur(0)
    , shaderFinal(0)
    , shaderDebug(0)
    , quadVAO(0)
    , quadVBO(0)
    , cudaDepth1Resource(nullptr)
    , cudaDepth2Resource(nullptr)
    , depthSurface1(0)
    , depthSurface2(0)
    , depthArray1(nullptr)
    , depthArray2(nullptr)
    , cudaInitialized(false)
{
    memset(boxsize, 0, sizeof(boxsize));
    memset(projMatrix, 0, sizeof(projMatrix));
}

FluidRenderer::~FluidRenderer() {
    cleanup();
}

void FluidRenderer::init(int width, int height, const float bs[3], float particleRadius, const float* proj) {
    windowWidth = width;
    windowHeight = height;
    boxsize[0] = bs[0];
    boxsize[1] = bs[1];
    boxsize[2] = bs[2];
    memcpy(projMatrix, proj, 16 * sizeof(float));
    
    // Extract projection parameters
    projW = projMatrix[0];
    projH = projMatrix[5];
    
    // Validate projection matrix is not zero
    if (projMatrix[10] == 0.0f) {
        std::cerr << "ERROR: Projection matrix not initialized! projMatrix[10] is 0.\n";
        // Use default values
        nearClip = 0.1f;
        farClip = 101.0f;
    } else {
        nearClip = projMatrix[14] / projMatrix[10];
        farClip = projMatrix[14] / (1.0f + projMatrix[10]);
    }
    
    std::cout << "FluidRenderer initialized:\n";
    std::cout << "  Window: " << width << "x" << height << "\n";
    std::cout << "  projW=" << projW << ", projH=" << projH << "\n";
    std::cout << "  nearClip=" << nearClip << ", farClip=" << farClip << "\n";
    
    // Initialize shaders
    initShaders();
    
    // Create fullscreen quad
    float quadVertices[] = {
        -1.0f, -1.0f, 0.0f, 0.0f,
         1.0f, -1.0f, 1.0f, 0.0f,
         1.0f,  1.0f, 1.0f, 1.0f,
        -1.0f, -1.0f, 0.0f, 0.0f,
         1.0f,  1.0f, 1.0f, 1.0f,
        -1.0f,  1.0f, 0.0f, 1.0f,
    };
    
    glGenVertexArrays(1, &quadVAO);
    glGenBuffers(1, &quadVBO);
    
    glBindVertexArray(quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
    
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);
    
    glBindVertexArray(0);
    
    // Initialize render targets
    depthTarget1.init(width, height, RenderTarget::R32F);
    depthTarget2.init(width, height, RenderTarget::R32F);
    thicknessTarget1.init(width, height, RenderTarget::R32F);
    thicknessTarget2.init(width, height, RenderTarget::R32F);
    
    // Initialize CUDA
    initCUDA();
}

void FluidRenderer::initShaders() {
    shaderBall = createProgram(pointSpriteVertexShader, ballFragmentShader);
    shaderDepth = createProgram(pointSpriteVertexShader, depthFragmentShader);
    shaderThickness = createProgram(pointSpriteVertexShader, thicknessFragmentShader);
    shaderBlur = createProgram(fullscreenVertexShader, blurFragmentShader);
    shaderFinal = createProgram(fullscreenVertexShader, finalFragmentShader);
    shaderDebug = createProgram(fullscreenVertexShader, debugFragmentShader);
}

const char* FluidRenderer::getDebugModeName() const {
    switch (debugMode) {
        case DEBUG_NONE: return "None (Full Render)";
        case DEBUG_DEPTH_RAW: return "Depth (Raw)";
        case DEBUG_DEPTH_SMOOTHED: return "Depth (Smoothed)";
        case DEBUG_THICKNESS: return "Thickness";
        case DEBUG_NORMALS: return "Normals";
        default: return "Unknown";
    }
}

void FluidRenderer::initCUDA() {
    // Register depth textures with CUDA
    cudaGraphicsGLRegisterImage(&cudaDepth1Resource, depthTarget1.getTexture(),
                                 GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
    cudaGraphicsGLRegisterImage(&cudaDepth2Resource, depthTarget2.getTexture(),
                                 GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
    cudaInitialized = true;
}

void FluidRenderer::cleanupCUDA() {
    if (cudaInitialized) {
        if (depthSurface1) cudaDestroySurfaceObject(depthSurface1);
        if (depthSurface2) cudaDestroySurfaceObject(depthSurface2);
        if (cudaDepth1Resource) cudaGraphicsUnregisterResource(cudaDepth1Resource);
        if (cudaDepth2Resource) cudaGraphicsUnregisterResource(cudaDepth2Resource);
        cudaInitialized = false;
    }
}

void FluidRenderer::resize(int width, int height) {
    if (width == windowWidth && height == windowHeight) return;
    
    windowWidth = width;
    windowHeight = height;
    
    // Cleanup old CUDA resources
    cleanupCUDA();
    
    // Resize render targets
    depthTarget1.resize(width, height);
    depthTarget2.resize(width, height);
    thicknessTarget1.resize(width, height);
    thicknessTarget2.resize(width, height);
    
    // Re-register with CUDA
    initCUDA();
}

void FluidRenderer::cleanup() {
    cleanupCUDA();
    
    depthTarget1.cleanup();
    depthTarget2.cleanup();
    thicknessTarget1.cleanup();
    thicknessTarget2.cleanup();
    
    if (shaderBall) glDeleteProgram(shaderBall);
    if (shaderDepth) glDeleteProgram(shaderDepth);
    if (shaderThickness) glDeleteProgram(shaderThickness);
    if (shaderBlur) glDeleteProgram(shaderBlur);
    if (shaderFinal) glDeleteProgram(shaderFinal);
    if (shaderDebug) glDeleteProgram(shaderDebug);
    
    if (quadVAO) glDeleteVertexArrays(1, &quadVAO);
    if (quadVBO) glDeleteBuffers(1, &quadVBO);
}

void FluidRenderer::render(GLuint positionVBO, int particleCount, float particleRadius,
                           const Camera& camera, int width, int height) {
    switch (renderMode) {
        case RENDERMODE_POINTS:
            renderPoints(positionVBO, particleCount);
            break;
        case RENDERMODE_BALLS:
            renderBalls(positionVBO, particleCount, particleRadius, camera, width, height);
            break;
        case RENDERMODE_ADVANCED:
            renderAdvanced(positionVBO, particleCount, particleRadius, camera, width, height);
            break;
    }
}

void FluidRenderer::renderPoints(GLuint positionVBO, int particleCount) {
    glBindBuffer(GL_ARRAY_BUFFER, positionVBO);
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(3, GL_FLOAT, sizeof(float4), 0);
    glColor3f(0.2f, 0.5f, 1.0f);
    glDrawArrays(GL_POINTS, 0, particleCount);
    glDisableClientState(GL_VERTEX_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void FluidRenderer::renderPointSprites(GLuint shader, GLuint positionVBO, int particleCount,
                                        float radius, const Camera& camera, int width, int height, bool enableDepthTest) {
    glUseProgram(shader);
    
    // Get matrices from current OpenGL state
    float modelView[16], proj[16], mvp[16];
    glGetFloatv(GL_MODELVIEW_MATRIX, modelView);
    glGetFloatv(GL_PROJECTION_MATRIX, proj);
    
    // Compute MVP
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            mvp[i * 4 + j] = 0;
            for (int k = 0; k < 4; k++) {
                mvp[i * 4 + j] += proj[k * 4 + j] * modelView[i * 4 + k];
            }
        }
    }
    
    glUniformMatrix4fv(glGetUniformLocation(shader, "modelViewProj"), 1, GL_FALSE, mvp);
    glUniformMatrix4fv(glGetUniformLocation(shader, "modelView"), 1, GL_FALSE, modelView);
    glUniformMatrix4fv(glGetUniformLocation(shader, "proj"), 1, GL_FALSE, proj);
    glUniform1f(glGetUniformLocation(shader, "radius"), radius);
    glUniform1f(glGetUniformLocation(shader, "pointScale"), height * proj[5]);
    glUniform1f(glGetUniformLocation(shader, "nearClip"), nearClip);
    glUniform1f(glGetUniformLocation(shader, "farClip"), farClip);
    
    if (enableDepthTest) {
        glEnable(GL_DEPTH_TEST);
        glDepthMask(GL_TRUE);
    } else {
        glDisable(GL_DEPTH_TEST);
    }
    
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
    glEnable(GL_POINT_SPRITE);
    
    glBindBuffer(GL_ARRAY_BUFFER, positionVBO);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);
    
    glDrawArrays(GL_POINTS, 0, particleCount);
    
    glDisableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDisable(GL_POINT_SPRITE);
    glUseProgram(0);
}

void FluidRenderer::renderBalls(GLuint positionVBO, int particleCount, float radius,
                                 const Camera& camera, int width, int height) {
    renderPointSprites(shaderBall, positionVBO, particleCount, radius, camera, width, height, true);
}

void FluidRenderer::renderFullscreenQuad(GLuint texture, int textureUnit) {
    glActiveTexture(GL_TEXTURE0 + textureUnit);
    glBindTexture(GL_TEXTURE_2D, texture);
    
    glBindVertexArray(quadVAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
    
    glBindTexture(GL_TEXTURE_2D, 0);
}

void FluidRenderer::smoothDepthTexture() {
    if (!cudaInitialized || smoothingIterations == 0) return;
    
    // Map textures to CUDA
    cudaGraphicsResource* resources[] = { cudaDepth1Resource, cudaDepth2Resource };
    cudaGraphicsMapResources(2, resources, 0);
    
    cudaGraphicsSubResourceGetMappedArray(&depthArray1, cudaDepth1Resource, 0, 0);
    cudaGraphicsSubResourceGetMappedArray(&depthArray2, cudaDepth2Resource, 0, 0);
    
    // Create surface objects
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    
    resDesc.res.array.array = depthArray1;
    cudaCreateSurfaceObject(&depthSurface1, &resDesc);
    
    resDesc.res.array.array = depthArray2;
    cudaCreateSurfaceObject(&depthSurface2, &resDesc);
    
    // Run curvature flow iterations
    for (int i = 0; i < smoothingIterations; i++) {
        launchCurvatureFlow(depthSurface1, depthSurface2,
                           windowWidth, windowHeight, smoothingDt, smoothingZContrib,
                           projW, projH);
        launchCurvatureFlow(depthSurface2, depthSurface1,
                           windowWidth, windowHeight, smoothingDt, smoothingZContrib,
                           projW, projH);
    }
    
    // Cleanup surface objects
    cudaDestroySurfaceObject(depthSurface1);
    cudaDestroySurfaceObject(depthSurface2);
    depthSurface1 = 0;
    depthSurface2 = 0;
    
    cudaGraphicsUnmapResources(2, resources, 0);
}

void FluidRenderer::renderAdvanced(GLuint positionVBO, int particleCount, float radius,
                                    const Camera& camera, int width, int height) {
    static bool firstFrame = true;
    
    // Pass 1: Render thickness map with additive blending
    thicknessTarget1.bind();
    thicknessTarget1.clear();
    
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE);
    renderPointSprites(shaderThickness, positionVBO, particleCount, radius * 3.0f, camera, width, height, false);
    glDisable(GL_BLEND);
    
    thicknessTarget1.unbind();
    
    if (firstFrame) checkGLError("after thickness pass");
    
    // DEBUG: Show thickness map
    if (debugMode == DEBUG_THICKNESS) {
        glDisable(GL_DEPTH_TEST);
        glUseProgram(shaderDebug);
        glUniform1i(glGetUniformLocation(shaderDebug, "tex"), 0);
        glUniform1i(glGetUniformLocation(shaderDebug, "vizMode"), 1);  // thickness mode
        renderFullscreenQuad(thicknessTarget1.getTexture());
        glUseProgram(0);
        return;
    }
    
    // Pass 1b: Blur thickness map (if enabled)
    if (blurThickness) {
        // Vertical pass
        thicknessTarget2.bind();
        thicknessTarget2.clear();
        
        glUseProgram(shaderBlur);
        glUniform1i(glGetUniformLocation(shaderBlur, "tex"), 0);
        glUniform2f(glGetUniformLocation(shaderBlur, "direction"), 0.0f, 1.0f);
        glUniform1f(glGetUniformLocation(shaderBlur, "texelSize"), 1.0f / windowHeight);
        
        glDisable(GL_DEPTH_TEST);
        renderFullscreenQuad(thicknessTarget1.getTexture());
        glUseProgram(0);
        
        thicknessTarget2.unbind();
        
        // Horizontal pass
        thicknessTarget1.bind();
        thicknessTarget1.clear();
        
        glUseProgram(shaderBlur);
        glUniform1i(glGetUniformLocation(shaderBlur, "tex"), 0);
        glUniform2f(glGetUniformLocation(shaderBlur, "direction"), 1.0f, 0.0f);
        glUniform1f(glGetUniformLocation(shaderBlur, "texelSize"), 1.0f / windowWidth);
        
        renderFullscreenQuad(thicknessTarget2.getTexture());
        glUseProgram(0);
        
        thicknessTarget1.unbind();
    }
    
    // Pass 2: Render depth map
    depthTarget1.bind();
    depthTarget1.clear();
    
    renderPointSprites(shaderDepth, positionVBO, particleCount, radius * 1.5f, camera, width, height, true);
    
    depthTarget1.unbind();
    
    if (firstFrame) checkGLError("after depth pass");
    
    // DEBUG: Show raw depth map (before smoothing)
    if (debugMode == DEBUG_DEPTH_RAW) {
        glDisable(GL_DEPTH_TEST);
        glUseProgram(shaderDebug);
        glUniform1i(glGetUniformLocation(shaderDebug, "tex"), 0);
        glUniform1i(glGetUniformLocation(shaderDebug, "vizMode"), 0);  // depth mode
        renderFullscreenQuad(depthTarget1.getTexture());
        glUseProgram(0);
        return;
    }
    
    // Pass 3: Smooth depth (CUDA curvature flow)
    if (smoothDepth) {
        smoothDepthTexture();
        if (firstFrame) {
            cudaError_t cudaErr = cudaGetLastError();
            if (cudaErr != cudaSuccess) {
                std::cerr << "CUDA error after smoothing: " << cudaGetErrorString(cudaErr) << std::endl;
            }
        }
    }
    
    if (firstFrame) checkGLError("after CUDA smoothing");
    
    // DEBUG: Show smoothed depth map
    if (debugMode == DEBUG_DEPTH_SMOOTHED) {
        glDisable(GL_DEPTH_TEST);
        glUseProgram(shaderDebug);
        glUniform1i(glGetUniformLocation(shaderDebug, "tex"), 0);
        glUniform1i(glGetUniformLocation(shaderDebug, "vizMode"), 0);  // depth mode
        renderFullscreenQuad(depthTarget1.getTexture());
        glUseProgram(0);
        return;
    }
    
    // Pass 4: Final compositing
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_DEPTH_TEST);
    
    glUseProgram(shaderFinal);
    
    // Get current matrices
    float modelView[16], proj[16];
    glGetFloatv(GL_MODELVIEW_MATRIX, modelView);
    glGetFloatv(GL_PROJECTION_MATRIX, proj);
    
    glUniform1i(glGetUniformLocation(shaderFinal, "depthTex"), 0);
    glUniform1i(glGetUniformLocation(shaderFinal, "thicknessTex"), 1);
    glUniformMatrix4fv(glGetUniformLocation(shaderFinal, "proj"), 1, GL_FALSE, proj);
    glUniformMatrix4fv(glGetUniformLocation(shaderFinal, "modelView"), 1, GL_FALSE, modelView);
    glUniform1f(glGetUniformLocation(shaderFinal, "nearClip"), nearClip);
    glUniform1f(glGetUniformLocation(shaderFinal, "farClip"), farClip);
    glUniform1f(glGetUniformLocation(shaderFinal, "radius"), radius);
    glUniform1f(glGetUniformLocation(shaderFinal, "projW"), projW);
    glUniform1f(glGetUniformLocation(shaderFinal, "projH"), projH);
    glUniform2f(glGetUniformLocation(shaderFinal, "windowSize"), (float)windowWidth, (float)windowHeight);
    
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, depthTarget1.getTexture());
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, thicknessTarget1.getTexture());
    
    glBindVertexArray(quadVAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
    
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, 0);
    
    glUseProgram(0);
    glDisable(GL_BLEND);
    
    if (firstFrame) {
        checkGLError("after final composite");
        firstFrame = false;
    }
}
