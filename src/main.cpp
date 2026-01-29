/**
 * PySPH C++ Port - Screen-Space Fluid Rendering Demo
 * 
 * A 1:1 port of the Python pysph demo using:
 * - CUDA (instead of OpenCL) for SPH simulation and curvature flow
 * - GLSL (instead of Cg) for shaders
 * - FreeGLUT + ImGui (instead of Qt) for windowing/UI
 */

#ifdef _WIN32
#include <Windows.h>
#endif
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <iostream>
#include <cmath>

#include <imgui.h>
#include <imgui_impl_opengl3.h>
#include <imgui_impl_glut.h>

#include "sph/SPHSimulator.h"
#include "rendering/FluidRenderer.h"
#include "Camera.h"

// Window dimensions
static int windowWidth = 800;
static int windowHeight = 800;

// Simulation
static SPHSimulator* sphSimulator = nullptr;
static FluidRenderer* fluidRenderer = nullptr;
static Camera camera;

// UI state
static bool showUI = true;
static bool paused = false;
static float fps = 0.0f;
static int frameCount = 0;
static float lastTime = 0.0f;

// Particle count (slider value and current)
static int desiredParticleCount = 8000;
static int currentParticleCount = 8000;

// GPU info
static std::string gpuName;

// Forward declarations
void display();
void reshape(int w, int h);
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void mouseWheel(int wheel, int direction, int x, int y);
void cleanup();
void resetSimulation();

void initGL() {
    // Gray background (matching Python: 150/255)
    glClearColor(150.0f/255.0f, 150.0f/255.0f, 150.0f/255.0f, 1.0f);
    glClearDepth(1.0f);
    glDepthFunc(GL_LESS);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
}

void renderBox() {
    // Render wireframe box (matching Python's render_box)
    glColor3f(0.0f, 0.0f, 0.0f);
    
    float boxsize[3];
    sphSimulator->getBoxSize(boxsize);
    
    glBegin(GL_LINES);
    // Bottom face
    glVertex3f(0, 0, 0); glVertex3f(boxsize[0], 0, 0);
    glVertex3f(boxsize[0], 0, 0); glVertex3f(boxsize[0], 0, boxsize[2]);
    glVertex3f(boxsize[0], 0, boxsize[2]); glVertex3f(0, 0, boxsize[2]);
    glVertex3f(0, 0, boxsize[2]); glVertex3f(0, 0, 0);
    // Top face
    glVertex3f(0, boxsize[1], 0); glVertex3f(boxsize[0], boxsize[1], 0);
    glVertex3f(boxsize[0], boxsize[1], 0); glVertex3f(boxsize[0], boxsize[1], boxsize[2]);
    glVertex3f(boxsize[0], boxsize[1], boxsize[2]); glVertex3f(0, boxsize[1], boxsize[2]);
    glVertex3f(0, boxsize[1], boxsize[2]); glVertex3f(0, boxsize[1], 0);
    // Vertical edges
    glVertex3f(0, 0, 0); glVertex3f(0, boxsize[1], 0);
    glVertex3f(boxsize[0], 0, 0); glVertex3f(boxsize[0], boxsize[1], 0);
    glVertex3f(boxsize[0], 0, boxsize[2]); glVertex3f(boxsize[0], boxsize[1], boxsize[2]);
    glVertex3f(0, 0, boxsize[2]); glVertex3f(0, boxsize[1], boxsize[2]);
    glEnd();
}

void display() {
    // Calculate delta time
    float currentTime = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;
    static float lastFrameTime = 0.0f;
    float deltaTime = currentTime - lastFrameTime;
    lastFrameTime = currentTime;
    
    // Update FPS
    frameCount++;
    if (currentTime - lastTime >= 1.0f) {
        fps = frameCount / (currentTime - lastTime);
        frameCount = 0;
        lastTime = currentTime;
    }
    
    // Run simulation steps
    if (!paused && sphSimulator) {
        // Match Python: run enough steps to simulate real-time at target framerate
        float dt = sphSimulator->getTimestep();
        int stepsPerFrame = (int)(1.0f / (60.0f * dt));
        for (int i = 0; i < stepsPerFrame; i++) {
            sphSimulator->step();
        }
    }
    
    // Clear
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    // Setup camera
    camera.applyProjection(windowWidth, windowHeight);
    camera.applyModelView();
    
    // Render box
    renderBox();
    
    // Reset modelview for particle rendering (camera will be reapplied in renderer)
    glLoadIdentity();
    camera.applyModelView();
    
    // Render fluid
    if (fluidRenderer && sphSimulator) {
        fluidRenderer->render(
            sphSimulator->getPositionVBO(),
            sphSimulator->getParticleCount(),
            sphSimulator->getParticleRadius(),
            camera,
            windowWidth, windowHeight
        );
    }
    
    // Render ImGui
    if (showUI) {
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGLUT_NewFrame();
        ImGui::NewFrame();
        
        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(300, 400), ImGuiCond_FirstUseEver);
        ImGui::Begin("SPH Fluid Demo", &showUI);
        
        ImGui::Text("FPS: %.1f", fps);
        ImGui::Text("GPU: %s", gpuName.c_str());
        ImGui::Text("Particles: %d", currentParticleCount);
        
        ImGui::Separator();
        ImGui::Text("Simulation:");
        
        if (ImGui::Checkbox("Paused", &paused)) {}
        
        ImGui::SliderInt("Particle Count", &desiredParticleCount, 270, 100000);
        if (desiredParticleCount != currentParticleCount) {
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "(pending)");
        }
        
        if (ImGui::Button("Reset Simulation (R)")) {
            resetSimulation();
        }
        
        ImGui::Separator();
        ImGui::Text("Render Mode:");
        
        if (fluidRenderer) {
            int mode = fluidRenderer->getRenderMode();
            if (ImGui::RadioButton("Points", mode == FluidRenderer::RENDERMODE_POINTS)) {
                fluidRenderer->setRenderMode(FluidRenderer::RENDERMODE_POINTS);
            }
            if (ImGui::RadioButton("Balls", mode == FluidRenderer::RENDERMODE_BALLS)) {
                fluidRenderer->setRenderMode(FluidRenderer::RENDERMODE_BALLS);
            }
            if (ImGui::RadioButton("Advanced", mode == FluidRenderer::RENDERMODE_ADVANCED)) {
                fluidRenderer->setRenderMode(FluidRenderer::RENDERMODE_ADVANCED);
            }
            
            if (mode == FluidRenderer::RENDERMODE_ADVANCED) {
                ImGui::Separator();
                ImGui::Text("Advanced Rendering:");
                
                bool smoothDepth = fluidRenderer->getSmoothDepth();
                if (ImGui::Checkbox("Smooth Depth", &smoothDepth)) {
                    fluidRenderer->setSmoothDepth(smoothDepth);
                }
                
                bool blurThickness = fluidRenderer->getBlurThickness();
                if (ImGui::Checkbox("Blur Thickness", &blurThickness)) {
                    fluidRenderer->setBlurThickness(blurThickness);
                }
                
                int iterations = fluidRenderer->getSmoothingIterations();
                if (ImGui::SliderInt("Smoothing Iterations", &iterations, 0, 100)) {
                    fluidRenderer->setSmoothingIterations(iterations);
                }
                
                float zContrib = fluidRenderer->getSmoothingZContrib();
                if (ImGui::SliderFloat("Z Contribution", &zContrib, 0.0f, 50.0f)) {
                    fluidRenderer->setSmoothingZContrib(zContrib);
                }
                
                ImGui::Separator();
                ImGui::Text("Debug Visualization:");
                int debugMode = fluidRenderer->getDebugMode();
                if (ImGui::RadioButton("None (Full Render)", debugMode == FluidRenderer::DEBUG_NONE)) {
                    fluidRenderer->setDebugMode(FluidRenderer::DEBUG_NONE);
                }
                if (ImGui::RadioButton("Depth (Raw)", debugMode == FluidRenderer::DEBUG_DEPTH_RAW)) {
                    fluidRenderer->setDebugMode(FluidRenderer::DEBUG_DEPTH_RAW);
                }
                if (ImGui::RadioButton("Depth (Smoothed)", debugMode == FluidRenderer::DEBUG_DEPTH_SMOOTHED)) {
                    fluidRenderer->setDebugMode(FluidRenderer::DEBUG_DEPTH_SMOOTHED);
                }
                if (ImGui::RadioButton("Thickness", debugMode == FluidRenderer::DEBUG_THICKNESS)) {
                    fluidRenderer->setDebugMode(FluidRenderer::DEBUG_THICKNESS);
                }
                ImGui::Text("Press D to cycle debug modes");
            }
        }
        
        ImGui::Separator();
        ImGui::Text("Controls:");
        ImGui::Text("  Left drag: Rotate");
        ImGui::Text("  Right drag: Translate");
        ImGui::Text("  Wheel: Zoom");
        ImGui::Text("  H: Toggle UI");
        ImGui::Text("  Space: Pause/Resume");
        ImGui::Text("  M: Cycle render mode");
        ImGui::Text("  D: Cycle debug mode");
        ImGui::Text("  R: Reset");
        
        ImGui::End();
        
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    }
    
    glutSwapBuffers();
    glutPostRedisplay();
}

void reshape(int w, int h) {
    if (h == 0) h = 1;
    windowWidth = w;
    windowHeight = h;
    glViewport(0, 0, w, h);
    ImGui_ImplGLUT_ReshapeFunc(w, h);
    
    // Resize render targets
    if (fluidRenderer) {
        fluidRenderer->resize(w, h);
    }
}

void keyboard(unsigned char key, int x, int y) {
    if (ImGui::GetCurrentContext()) {
        ImGuiIO& io = ImGui::GetIO();
        if (io.WantCaptureKeyboard) return;
    }
    
    switch (key) {
        case 'h':
        case 'H':
            showUI = !showUI;
            break;
        case ' ':
            paused = !paused;
            break;
        case 'r':
        case 'R':
            resetSimulation();
            break;
        case 'm':
        case 'M':
            if (fluidRenderer) {
                fluidRenderer->cycleRenderMode();
            }
            break;
        case 'd':
        case 'D':
            if (fluidRenderer) {
                fluidRenderer->cycleDebugMode();
            }
            break;
        case 27: // ESC
            cleanup();
            exit(0);
            break;
    }
}

// Mouse state
static int lastMouseX = 0;
static int lastMouseY = 0;
static bool leftButtonDown = false;
static bool rightButtonDown = false;

void mouse(int button, int state, int x, int y) {
    if (ImGui::GetCurrentContext()) {
        ImGuiIO& io = ImGui::GetIO();
        ImGui_ImplGLUT_MouseFunc(button, state, x, y);
        if (io.WantCaptureMouse) return;
    }
    
    if (button == GLUT_LEFT_BUTTON) {
        leftButtonDown = (state == GLUT_DOWN);
        lastMouseX = x;
        lastMouseY = y;
    }
    else if (button == GLUT_RIGHT_BUTTON) {
        rightButtonDown = (state == GLUT_DOWN);
        lastMouseX = x;
        lastMouseY = y;
    }
}

void motion(int x, int y) {
    if (ImGui::GetCurrentContext()) {
        ImGuiIO& io = ImGui::GetIO();
        ImGui_ImplGLUT_MotionFunc(x, y);
        if (io.WantCaptureMouse) return;
    }
    
    int dx = x - lastMouseX;
    int dy = y - lastMouseY;
    lastMouseX = x;
    lastMouseY = y;
    
    if (leftButtonDown) {
        // Rotate
        camera.rotate(dx * 0.2f, dy * 0.2f);
    }
    else if (rightButtonDown) {
        // Translate
        camera.translate(dx * 0.03f, -dy * 0.03f);
    }
}

void mouseWheel(int wheel, int direction, int x, int y) {
    if (ImGui::GetCurrentContext()) {
        ImGuiIO& io = ImGui::GetIO();
        if (io.WantCaptureMouse) {
            io.MouseWheel += (float)direction;
            return;
        }
    }
    
    camera.zoom(direction * 0.5f);
}

void cleanup() {
    if (fluidRenderer) {
        fluidRenderer->cleanup();
        delete fluidRenderer;
        fluidRenderer = nullptr;
    }
    
    if (sphSimulator) {
        sphSimulator->cleanup();
        delete sphSimulator;
        sphSimulator = nullptr;
    }
    
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGLUT_Shutdown();
    ImGui::DestroyContext();
}

void resetSimulation() {
    // Use the desired particle count from the slider
    int N = desiredParticleCount;
    if (N < 270) N = 270;
    
    // Cleanup old simulator
    if (sphSimulator) {
        sphSimulator->cleanup();
        delete sphSimulator;
        sphSimulator = nullptr;
    }
    
    // Create new simulator with updated particle count
    float boxsize[3] = {10.0f, 10.0f, 10.0f};
    sphSimulator = new SPHSimulator(N, boxsize);
    sphSimulator->init();
    
    currentParticleCount = sphSimulator->getParticleCount();
    
    // Update renderer with new particle radius
    if (fluidRenderer) {
        // Renderer doesn't need recreation, just reset camera
    }
    
    camera.reset();
    
    std::cout << "Simulation reset with " << currentParticleCount << " particles\n";
}

int main(int argc, char** argv) {
    std::cout << "PySPH C++ Port - Screen-Space Fluid Rendering\n";
    std::cout << "==============================================\n\n";
    
    // Parse command line for particle count
    int N = 8000;
    if (argc > 1) {
        N = atoi(argv[1]);
        if (N < 270) {
            std::cerr << "N must be at least 270\n";
            return 1;
        }
    } else {
        std::cout << "Usage: pysph-cpp [num_particles]\n";
        std::cout << "Using default: " << N << " particles\n\n";
    }
    desiredParticleCount = N;
    
    // Initialize GLUT
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    glutInitWindowSize(windowWidth, windowHeight);
    glutCreateWindow("PySPH C++ - Fluid Simulation");
    
    // Initialize GLEW
    GLenum err = glewInit();
    if (err != GLEW_OK) {
        std::cerr << "GLEW Error: " << glewGetErrorString(err) << std::endl;
        return -1;
    }
    
    // Get GPU info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    gpuName = prop.name;
    std::cout << "GPU: " << gpuName << "\n\n";
    
    // Initialize OpenGL
    initGL();
    
    // Initialize ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplGLUT_Init();
    ImGui_ImplOpenGL3_Init("#version 130");
    ImGui::StyleColorsDark();
    
    // Initialize camera (matching Python: initrans = [-6, -4, -20])
    camera.init(-6.0f, -4.0f, -20.0f);
    // IMPORTANT: Apply projection to compute the projection matrix BEFORE passing to renderer
    camera.applyProjection(windowWidth, windowHeight);
    
    // Create SPH simulator
    float boxsize[3] = {10.0f, 10.0f, 10.0f};
    sphSimulator = new SPHSimulator(N, boxsize);
    sphSimulator->init();
    currentParticleCount = sphSimulator->getParticleCount();
    
    // Create fluid renderer
    fluidRenderer = new FluidRenderer();
    fluidRenderer->init(windowWidth, windowHeight, boxsize, 
                        sphSimulator->getParticleRadius(),
                        camera.getProjectionMatrix());
    
    // Setup callbacks
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutPassiveMotionFunc(ImGui_ImplGLUT_MotionFunc);
    glutMouseWheelFunc(mouseWheel);
    
    std::cout << "Controls:\n";
    std::cout << "  Left drag: Rotate view\n";
    std::cout << "  Right drag: Translate view\n";
    std::cout << "  Mouse wheel: Zoom\n";
    std::cout << "  H: Toggle UI\n";
    std::cout << "  Space: Pause/Resume\n";
    std::cout << "  M: Cycle render mode\n";
    std::cout << "  R: Reset simulation\n";
    std::cout << "  ESC: Exit\n\n";
    
    // Main loop
    glutMainLoop();
    
    cleanup();
    return 0;
}
