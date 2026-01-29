#pragma once

#ifdef _WIN32
#include <Windows.h>
#endif
#include <GL/glew.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "RenderTarget.h"
#include "../Camera.h"

/**
 * Screen-space fluid renderer - Port of Python pysph FluidRenderer.
 * 
 * Implements Simon Green's screen-space fluid rendering technique:
 * 1. Render particle depths to texture
 * 2. Smooth depth using curvature flow
 * 3. Render thickness map with additive blending
 * 4. Blur thickness map
 * 5. Final compositing with Beer's law and Fresnel
 */
class FluidRenderer {
public:
    static const int RENDERMODE_POINTS = 0;
    static const int RENDERMODE_BALLS = 1;
    static const int RENDERMODE_ADVANCED = 2;
    
    FluidRenderer();
    ~FluidRenderer();
    
    void init(int width, int height, const float boxsize[3], float particleRadius, const float* projMatrix);
    void resize(int width, int height);
    void cleanup();
    
    void render(GLuint positionVBO, int particleCount, float particleRadius,
                const Camera& camera, int width, int height);
    
    // Render mode
    void setRenderMode(int mode) { renderMode = mode; }
    int getRenderMode() const { return renderMode; }
    void cycleRenderMode() { renderMode = (renderMode + 1) % 3; }
    
    // Smoothing parameters
    void setSmoothDepth(bool enabled) { smoothDepth = enabled; }
    bool getSmoothDepth() const { return smoothDepth; }
    
    void setBlurThickness(bool enabled) { blurThickness = enabled; }
    bool getBlurThickness() const { return blurThickness; }
    
    void setSmoothingIterations(int iters) { smoothingIterations = iters; }
    int getSmoothingIterations() const { return smoothingIterations; }
    
    void setSmoothingZContrib(float zc) { smoothingZContrib = zc; }
    float getSmoothingZContrib() const { return smoothingZContrib; }
    
    // Debug visualization modes
    enum DebugMode {
        DEBUG_NONE = 0,
        DEBUG_DEPTH_RAW,        // Depth map before smoothing
        DEBUG_DEPTH_SMOOTHED,   // Depth map after smoothing
        DEBUG_THICKNESS,        // Thickness map
        DEBUG_NORMALS,          // Computed normals
        DEBUG_COUNT
    };
    void setDebugMode(int mode) { debugMode = mode % DEBUG_COUNT; }
    int getDebugMode() const { return debugMode; }
    void cycleDebugMode() { debugMode = (debugMode + 1) % DEBUG_COUNT; }
    const char* getDebugModeName() const;
    
private:
    void initShaders();
    void initCUDA();
    void cleanupCUDA();
    
    void renderPoints(GLuint positionVBO, int particleCount);
    void renderBalls(GLuint positionVBO, int particleCount, float radius, const Camera& camera, int width, int height);
    void renderAdvanced(GLuint positionVBO, int particleCount, float radius, const Camera& camera, int width, int height);
    
    void renderPointSprites(GLuint shader, GLuint positionVBO, int particleCount, 
                            float radius, const Camera& camera, int width, int height, bool enableDepthTest);
    void renderFullscreenQuad(GLuint texture, int textureUnit = 0);
    void smoothDepthTexture();
    
    // Window size
    int windowWidth;
    int windowHeight;
    
    // Scene parameters
    float boxsize[3];
    float projMatrix[16];
    float nearClip, farClip;
    float projW, projH;  // Projection matrix diagonal elements
    
    // Render mode
    int renderMode;
    
    // Smoothing parameters (matching Python defaults)
    bool smoothDepth;
    bool blurThickness;
    int smoothingIterations;
    float smoothingDt;
    float smoothingZContrib;
    
    // Debug mode
    int debugMode;
    
    // Render targets
    RenderTarget depthTarget1;
    RenderTarget depthTarget2;
    RenderTarget thicknessTarget1;
    RenderTarget thicknessTarget2;
    
    // Shaders
    GLuint shaderBall;       // Lit sphere point sprites
    GLuint shaderDepth;      // Eye-space depth
    GLuint shaderThickness;  // Additive thickness
    GLuint shaderBlur;       // Gaussian blur
    GLuint shaderFinal;      // Compositing
    GLuint shaderDebug;      // Debug visualization
    
    // Fullscreen quad VAO
    GLuint quadVAO;
    GLuint quadVBO;
    
    // CUDA resources for depth smoothing
    cudaGraphicsResource* cudaDepth1Resource;
    cudaGraphicsResource* cudaDepth2Resource;
    cudaSurfaceObject_t depthSurface1;
    cudaSurfaceObject_t depthSurface2;
    cudaArray_t depthArray1;
    cudaArray_t depthArray2;
    
    bool cudaInitialized;
};
