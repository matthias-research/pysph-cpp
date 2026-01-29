#include "Camera.h"
#ifdef _WIN32
#include <Windows.h>
#endif
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cstring>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

Camera::Camera() 
    : rotateX(0), rotateY(0)
    , translateX(0), translateY(0), translateZ(0)
    , fovY((float)(M_PI / 2.5))  // Matching Python
    , nearClip(0.1f)
    , farClip(101.0f)
{
    initTrans[0] = 0;
    initTrans[1] = 0;
    initTrans[2] = -5;
    memset(projectionMatrix, 0, sizeof(projectionMatrix));
}

void Camera::init(float initX, float initY, float initZ) {
    initTrans[0] = initX;
    initTrans[1] = initY;
    initTrans[2] = initZ;
    reset();
}

void Camera::reset() {
    rotateX = 0;
    rotateY = 0;
    translateX = 0;
    translateY = 0;
    translateZ = 0;
}

void Camera::updateProjectionMatrix(float fovY, float aspect, float zNear, float zFar) {
    // Right-handed perspective projection (matching Python)
    float h = 1.0f / tanf(fovY * 0.5f);
    float w = h / aspect;
    
    // Column-major order for OpenGL
    memset(projectionMatrix, 0, sizeof(projectionMatrix));
    projectionMatrix[0] = w;
    projectionMatrix[5] = h;
    projectionMatrix[10] = zFar / (zNear - zFar);
    projectionMatrix[11] = -1.0f;
    projectionMatrix[14] = zNear * zFar / (zNear - zFar);
    // projectionMatrix[15] = 0
}

void Camera::applyProjection(int width, int height) {
    float aspect = (float)width / (float)height;
    updateProjectionMatrix(fovY, aspect, nearClip, farClip);
    
    glMatrixMode(GL_PROJECTION);
    glLoadMatrixf(projectionMatrix);
    glMatrixMode(GL_MODELVIEW);
}

void Camera::applyModelView() {
    // Matching Python's mouse_transform():
    // 1. Translate by initrans
    // 2. Rotate
    // 3. Translate by user offset
    
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    
    // Initial translation (camera position)
    glTranslatef(initTrans[0], initTrans[1], initTrans[2]);
    
    // Rotation
    glRotatef(rotateX, 1, 0, 0);
    glRotatef(rotateY, 0, 1, 0);
    
    // User translation
    glTranslatef(translateX, translateY, translateZ);
}

void Camera::rotate(float dx, float dy) {
    rotateX += dy;
    rotateY += dx;
}

void Camera::translate(float dx, float dy) {
    translateX += dx;
    translateY += dy;
}

void Camera::zoom(float delta) {
    translateZ += delta;
}
