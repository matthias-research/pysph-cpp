#include "SPHSimulator.h"
#ifdef _WIN32
#include <Windows.h>
#endif
#include <iostream>
#include <cmath>
#include <cstring>
#include <cstdlib>

// CUDA kernel declarations (implemented in SPHKernels.cu)
extern void launchComputeHash(float4* position, unsigned int* gridHash, unsigned int* gridIndex,
                              int N, int3 numCells, float3 boxsize, float h);
extern void launchFindCellStartEnd(unsigned int* cellStart, unsigned int* cellEnd,
                                   unsigned int* gridHash, unsigned int* gridIndex,
                                   float4* position, float4* positionSorted,
                                   float4* velocity, float4* velocitySorted,
                                   int N, int p2, int totalCells, bool reorder);
extern void launchStepDensity(float4* position, float* density, float* pressure,
                              unsigned int* gridIndex, unsigned int* cellStart, unsigned int* cellEnd,
                              SPHParams params, bool useGrid, bool reorder);
extern void launchStepForces(float4* position, float4* velocity, float4* acceleration,
                             float* density, float* pressure,
                             unsigned int* gridIndex, unsigned int* cellStart, unsigned int* cellEnd,
                             SPHParams params, bool useGrid, bool reorder);
extern void launchStepMove(float4* position, float4* velocity, float4* acceleration,
                           SPHParams params);
extern void thrustSortByKey(unsigned int* keys, unsigned int* values, int count);

SPHSimulator::SPHSimulator(int requestedN, const float bs[3])
    : N(requestedN)
    , p2(1)
    , k(1000.0f)
    , viscosity(250.0f)
    , density0(1000.0f)
    , wgSize(64)
    , positionVBO(0)
    , cudaVBOResource(nullptr)
    , h_position(nullptr)
    , h_velocity(nullptr)
    , d_position(nullptr)
    , d_positionSorted(nullptr)
    , d_velocity(nullptr)
    , d_velocitySorted(nullptr)
    , d_acceleration(nullptr)
    , d_density(nullptr)
    , d_pressure(nullptr)
    , d_gridHash(nullptr)
    , d_gridIndex(nullptr)
    , d_cellStart(nullptr)
    , d_cellEnd(nullptr)
    , initialized(false)
{
    boxsize[0] = bs[0];
    boxsize[1] = bs[1];
    boxsize[2] = bs[2];
    
    // Match Python: adjust N to have the right number of particles for cube arrangement
    // N = ((a-2)^3 * 5) / 4 where a is the largest even number such that a^3 * 1.25 <= N
    int a = 0;
    while ((a * a * a) * 1.25f <= requestedN) {
        a += 2;
    }
    N = ((a - 2) * (a - 2) * (a - 2) * 5) / 4;
    
    // p2 = smallest power of 2 >= N
    while (p2 < N) p2 *= 2;
    
    // Calculate spacing and timestep
    float maxBox = fmaxf(boxsize[0], fmaxf(boxsize[1], boxsize[2]));
    spacing0 = 0.5f * maxBox / powf((float)N, 1.0f/3.0f);
    dt = 0.5f * spacing0 / sqrtf(k);
    h = 2.000001f * spacing0;
    mass = spacing0 * spacing0 * spacing0 * density0;
    
    // Grid cells (1 cell per kernel radius)
    numCells[0] = (int)ceilf(boxsize[0] / h);
    numCells[1] = (int)ceilf(boxsize[1] / h);
    numCells[2] = (int)ceilf(boxsize[2] / h);
    totalCells = numCells[0] * numCells[1] * numCells[2];
    
    std::cout << "SPH Simulator:\n";
    std::cout << "  Particles: " << N << " (p2=" << p2 << ")\n";
    std::cout << "  Spacing: " << spacing0 << ", h: " << h << ", dt: " << dt << "\n";
    std::cout << "  Mass: " << mass << ", k: " << k << ", viscosity: " << viscosity << "\n";
    std::cout << "  Grid: " << numCells[0] << "x" << numCells[1] << "x" << numCells[2] 
              << " = " << totalCells << " cells\n";
}

SPHSimulator::~SPHSimulator() {
    cleanup();
}

void SPHSimulator::initializePositions() {
    // Match Python's initialize_positions()
    // Arrange particles in one large cube and two small cubes
    
    int n = (int)(powf((float)N / 1.25f, 1.0f/3.0f) + 0.5f);
    int i = 0;
    
    // Large cube
    for (int x = 0; x < n && i < N; x++) {
        for (int y = 0; y < n && i < N; y++) {
            for (int z = 0; z < n && i < N; z++) {
                h_position[i].x = (x + 0.5f) * spacing0;
                h_position[i].y = (y + 0.5f) * spacing0 + 0.01f * boxsize[1];
                h_position[i].z = boxsize[2] - (z + 0.5f) * spacing0;
                h_position[i].w = 0;
                i++;
            }
        }
    }
    
    // First small cube
    int halfN = n / 2;
    for (int x = 0; x < halfN && i < N; x++) {
        for (int y = 0; y < halfN && i < N; y++) {
            for (int z = 0; z < halfN && i < N; z++) {
                h_position[i].x = (x + 0.5f) * spacing0 + 0.01f * boxsize[0];
                h_position[i].y = boxsize[1] - (y + 0.5f) * spacing0;
                h_position[i].z = boxsize[2] - (z + 0.5f) * spacing0;
                h_position[i].w = 0;
                i++;
            }
        }
    }
    
    // Second small cube
    for (int x = 0; x < halfN && i < N; x++) {
        for (int y = 0; y < halfN && i < N; y++) {
            for (int z = 0; z < halfN && i < N; z++) {
                h_position[i].x = boxsize[0] - (x + 0.5f) * spacing0;
                h_position[i].y = (y + 0.5f) * spacing0 + 0.15f * boxsize[1];
                h_position[i].z = (z + 0.5f) * spacing0;
                h_position[i].w = 0;
                i++;
            }
        }
    }
    
    // Initialize velocities to zero
    memset(h_velocity, 0, N * sizeof(float4));
}

void SPHSimulator::createVBO() {
    // Create OpenGL VBO for positions
    glGenBuffers(1, &positionVBO);
    glBindBuffer(GL_ARRAY_BUFFER, positionVBO);
    glBufferData(GL_ARRAY_BUFFER, N * sizeof(float4), h_position, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    
    // Register with CUDA
    cudaGraphicsGLRegisterBuffer(&cudaVBOResource, positionVBO, cudaGraphicsMapFlagsNone);
}

void SPHSimulator::initCUDA() {
    // Allocate device memory
    cudaMalloc(&d_positionSorted, N * sizeof(float4));
    cudaMalloc(&d_velocity, N * sizeof(float4));
    cudaMalloc(&d_velocitySorted, N * sizeof(float4));
    cudaMalloc(&d_acceleration, N * sizeof(float4));
    cudaMalloc(&d_density, N * sizeof(float));
    cudaMalloc(&d_pressure, N * sizeof(float));
    
    // Grid arrays (p2 size for sorting)
    cudaMalloc(&d_gridHash, p2 * sizeof(unsigned int));
    cudaMalloc(&d_gridIndex, p2 * sizeof(unsigned int));
    cudaMalloc(&d_cellStart, totalCells * sizeof(unsigned int));
    cudaMalloc(&d_cellEnd, totalCells * sizeof(unsigned int));
    
    // Copy initial velocity to device
    cudaMemcpy(d_velocity, h_velocity, N * sizeof(float4), cudaMemcpyHostToDevice);
    
    // Initialize grid hash to max value (unused entries sort to end)
    cudaMemset(d_gridHash, 0xFF, p2 * sizeof(unsigned int));
}

void SPHSimulator::init() {
    // Allocate host memory
    h_position = new float4[N];
    h_velocity = new float4[N];
    
    // Initialize positions
    initializePositions();
    
    // Create VBO and register with CUDA
    createVBO();
    
    // Initialize CUDA resources
    initCUDA();
    
    initialized = true;
}

void SPHSimulator::assignCells() {
    // Map VBO
    size_t numBytes;
    cudaGraphicsMapResources(1, &cudaVBOResource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&d_position, &numBytes, cudaVBOResource);
    
    // Compute hash for each particle
    int3 nc = make_int3(numCells[0], numCells[1], numCells[2]);
    float3 bs = make_float3(boxsize[0], boxsize[1], boxsize[2]);
    launchComputeHash(d_position, d_gridHash, d_gridIndex, N, nc, bs, h);
    
    // Sort by hash using Thrust
    thrustSortByKey(d_gridHash, d_gridIndex, p2);
    
    // Find cell start/end and reorder particles
    launchFindCellStartEnd(d_cellStart, d_cellEnd, d_gridHash, d_gridIndex,
                           d_position, d_positionSorted, d_velocity, d_velocitySorted,
                           N, p2, totalCells, true);
    
    cudaGraphicsUnmapResources(1, &cudaVBOResource, 0);
}

void SPHSimulator::step() {
    if (!initialized) return;
    
    // Build grid structure
    assignCells();
    
    // Map VBO for simulation
    size_t numBytes;
    cudaGraphicsMapResources(1, &cudaVBOResource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&d_position, &numBytes, cudaVBOResource);
    
    // Setup parameters
    SPHParams params;
    params.N = N;
    params.dt = dt;
    params.mass = mass;
    params.h = h;
    params.h2 = h * h;
    params.k = k;
    params.viscosity = viscosity;
    params.density0 = density0;
    params.numCells = make_int3(numCells[0], numCells[1], numCells[2]);
    params.boxsize = make_float3(boxsize[0], boxsize[1], boxsize[2]);
    
    // Step 1: Compute density and pressure
    launchStepDensity(d_positionSorted, d_density, d_pressure,
                      d_gridIndex, d_cellStart, d_cellEnd,
                      params, true, true);
    
    // Step 2: Compute forces
    launchStepForces(d_positionSorted, d_velocitySorted, d_acceleration,
                     d_density, d_pressure,
                     d_gridIndex, d_cellStart, d_cellEnd,
                     params, true, true);
    
    // Step 3: Integrate (updates position VBO directly)
    launchStepMove(d_position, d_velocity, d_acceleration, params);
    
    cudaGraphicsUnmapResources(1, &cudaVBOResource, 0);
}

void SPHSimulator::reset() {
    if (!initialized) return;
    
    // Reinitialize positions
    initializePositions();
    
    // Upload to VBO
    glBindBuffer(GL_ARRAY_BUFFER, positionVBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, N * sizeof(float4), h_position);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    
    // Reset velocities
    memset(h_velocity, 0, N * sizeof(float4));
    cudaMemcpy(d_velocity, h_velocity, N * sizeof(float4), cudaMemcpyHostToDevice);
}

void SPHSimulator::cleanup() {
    if (!initialized) return;
    
    // Unregister VBO
    if (cudaVBOResource) {
        cudaGraphicsUnregisterResource(cudaVBOResource);
        cudaVBOResource = nullptr;
    }
    
    // Delete VBO
    if (positionVBO) {
        glDeleteBuffers(1, &positionVBO);
        positionVBO = 0;
    }
    
    // Free device memory
    cudaFree(d_positionSorted);
    cudaFree(d_velocity);
    cudaFree(d_velocitySorted);
    cudaFree(d_acceleration);
    cudaFree(d_density);
    cudaFree(d_pressure);
    cudaFree(d_gridHash);
    cudaFree(d_gridIndex);
    cudaFree(d_cellStart);
    cudaFree(d_cellEnd);
    
    // Free host memory
    delete[] h_position;
    delete[] h_velocity;
    h_position = nullptr;
    h_velocity = nullptr;
    
    initialized = false;
}
