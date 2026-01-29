#pragma once

#ifdef _WIN32
#include <Windows.h>
#endif
#include <GL/glew.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

/**
 * SPH Fluid Simulator - Port of Python pysph simulation.
 * 
 * Uses CUDA for GPU computation with OpenGL interop for rendering.
 * Implements grid-based neighbor search with Thrust sorting.
 */
class SPHSimulator {
public:
    SPHSimulator(int N, const float boxsize[3]);
    ~SPHSimulator();
    
    // Initialize CUDA resources and VBO
    void init();
    
    // Run one simulation step
    void step();
    
    // Reset simulation to initial state
    void reset();
    
    // Cleanup resources
    void cleanup();
    
    // Accessors
    GLuint getPositionVBO() const { return positionVBO; }
    int getParticleCount() const { return N; }
    float getTimestep() const { return dt; }
    float getParticleRadius() const { return h * 0.25f; }
    float getKernelRadius() const { return h; }
    void getBoxSize(float out[3]) const { out[0] = boxsize[0]; out[1] = boxsize[1]; out[2] = boxsize[2]; }
    
private:
    void initializePositions();
    void createVBO();
    void initCUDA();
    void assignCells();
    
    // Particle count
    int N;
    int p2;  // Smallest power of 2 >= N (for sorting)
    
    // Simulation parameters (matching Python)
    float boxsize[3];
    float k;           // Gas constant
    float viscosity;   // Fluid viscosity
    float spacing0;    // Initial particle spacing
    float dt;          // Timestep
    float h;           // Kernel support radius
    float density0;    // Rest density
    float mass;        // Particle mass
    
    // Grid parameters
    int numCells[3];
    int totalCells;
    int wgSize;        // Work group size for hashing
    
    // Host arrays (for initialization)
    float4* h_position;
    float4* h_velocity;
    
    // OpenGL VBO
    GLuint positionVBO;
    cudaGraphicsResource* cudaVBOResource;
    
    // Device arrays
    float4* d_position;         // Mapped from VBO
    float4* d_positionSorted;
    float4* d_velocity;
    float4* d_velocitySorted;
    float4* d_acceleration;
    float* d_density;
    float* d_pressure;
    
    // Grid arrays
    unsigned int* d_gridHash;
    unsigned int* d_gridIndex;
    unsigned int* d_cellStart;
    unsigned int* d_cellEnd;
    
    bool initialized;
};

#include "SPHParams.h"
