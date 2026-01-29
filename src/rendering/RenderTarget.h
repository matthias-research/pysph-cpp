#pragma once

#ifdef _WIN32
#include <Windows.h>
#endif
#include <GL/glew.h>

/**
 * Framebuffer Object wrapper for off-screen rendering.
 * Matches Python's RenderTarget and RenderTargetR32F classes.
 */
class RenderTarget {
public:
    enum Format {
        RGBA8,      // Standard RGBA
        R32F        // Single channel 32-bit float
    };
    
    RenderTarget();
    ~RenderTarget();
    
    void init(int width, int height, Format format = RGBA8);
    void resize(int width, int height);
    void cleanup();
    
    void bind();
    void unbind();
    
    // Clear with specified color (default: black with alpha 1)
    void clear(float r = 0.0f, float g = 0.0f, float b = 0.0f, float a = 1.0f);
    
    GLuint getTexture() const { return texture; }
    int getWidth() const { return width; }
    int getHeight() const { return height; }
    
private:
    GLuint fbo;
    GLuint texture;
    GLuint depthRbo;
    int width;
    int height;
    Format format;
    
    // Saved viewport for restore on unbind
    GLint savedViewport[4];
};
