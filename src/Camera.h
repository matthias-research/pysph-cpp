#pragma once

#include <cmath>

/**
 * Camera class matching Python's BaseDemo camera behavior.
 * Uses fixed-function OpenGL matrix operations.
 */
class Camera {
public:
    Camera();
    
    void init(float initX, float initY, float initZ);
    void reset();
    
    // Apply transformations
    void applyProjection(int width, int height);
    void applyModelView();
    
    // Mouse interaction
    void rotate(float dx, float dy);
    void translate(float dx, float dy);
    void zoom(float delta);
    
    // Get projection matrix (for passing to renderer)
    const float* getProjectionMatrix() const { return projectionMatrix; }
    
private:
    void updateProjectionMatrix(float fovY, float aspect, float nearClip, float farClip);
    
    // Initial translation (set once)
    float initTrans[3];
    
    // Current rotation (degrees)
    float rotateX;
    float rotateY;
    
    // Current translation offset
    float translateX;
    float translateY;
    float translateZ;
    
    // Projection parameters
    float fovY;
    float nearClip;
    float farClip;
    
    // Cached projection matrix (column-major for OpenGL)
    float projectionMatrix[16];
};
