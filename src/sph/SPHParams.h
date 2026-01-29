#pragma once

#include <cuda_runtime.h>

/**
 * SPH simulation parameters - shared between host and device code.
 * Separated from SPHSimulator.h to avoid OpenGL dependencies in CUDA files.
 */
struct SPHParams {
    int N;
    float dt;
    float mass;
    float h;
    float h2;
    float k;
    float viscosity;
    float density0;
    int3 numCells;
    float3 boxsize;
};
