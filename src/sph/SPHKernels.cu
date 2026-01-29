/**
 * SPH CUDA Kernels - Port of Python pysph sph.cl
 * 
 * Implements:
 * - Spatial hashing for neighbor search
 * - M4 (cubic spline) kernel for density
 * - Pressure and viscosity forces
 * - Position integration with boundary collision
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <cstdio>

#include "SPHParams.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// Block size for kernels
#define BLOCK_SIZE 128

//-----------------------------------------------------------------------------
// Device helper functions
//-----------------------------------------------------------------------------

__device__ inline float vlen(float3 x) {
    return sqrtf(x.x * x.x + x.y * x.y + x.z * x.z);
}

__device__ inline float3 make_float3_from_float4(float4 v) {
    return make_float3(v.x, v.y, v.z);
}

__device__ inline float4 make_float4_from_float3(float3 v) {
    return make_float4(v.x, v.y, v.z, 0.0f);
}

// M4 cubic spline kernel (matching Python's kernelM4)
__device__ inline float kernelM4(float x, float h) {
    float q = x / h;
    if (q >= 1.0f) return 0.0f;
    
    float factor = 2.546479089470325472f / (h * h * h);
    if (q < 0.5f) {
        return factor * (1.0f - 6.0f * q * q * (1.0f - q));
    }
    float a = 1.0f - q;
    return factor * 2.0f * a * a * a;
}

// Derivative of M4 kernel (matching Python's kernelM4_d)
__device__ inline float kernelM4_d(float x, float h) {
    float q = x / h;
    if (q >= 1.0f) return 0.0f;
    
    float h5 = h * h * h * h * h;
    float factor = 2.546479089470325472f / h5;
    if (q < 0.5f) {
        return factor * (-12.0f + 18.0f * q);
    }
    return factor * (-6.0f * (1.0f - q) * (1.0f - q) / q);
}

// Laplacian of viscosity kernel (matching Python's kernelVisc_dd)
__device__ inline float kernelVisc_dd(float x, float h) {
    float q = x / h;
    if (q <= 1.0f) {
        float h5 = h * h * h * h * h;
        return -45.0f / (M_PI * h5) * (1.0f - q);
    }
    return 0.0f;
}

// Grid position from world position
__device__ inline int3 getGridPosition(float3 position, int3 numCells, float3 boxsize, float h) {
    int x = min((int)(position.x / h), numCells.x - 1);
    int y = min((int)(position.y / h), numCells.y - 1);
    int z = min((int)(position.z / h), numCells.z - 1);
    return make_int3(max(0, x), max(0, y), max(0, z));
}

// Hash from grid position
__device__ inline unsigned int getGridHash(int3 gridPos, int3 numCells) {
    return (gridPos.z * numCells.y + gridPos.y) * numCells.x + gridPos.x;
}

//-----------------------------------------------------------------------------
// Compute hash kernel
//-----------------------------------------------------------------------------

__global__ void computeHashKernel(
    float4* position,
    unsigned int* gridHash,
    unsigned int* gridIndex,
    int N,
    int3 numCells,
    float3 boxsize,
    float h)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    float3 pos = make_float3_from_float4(position[idx]);
    int3 gridPos = getGridPosition(pos, numCells, boxsize, h);
    
    gridHash[idx] = getGridHash(gridPos, numCells);
    gridIndex[idx] = idx;
}

//-----------------------------------------------------------------------------
// Find cell start/end kernel
//-----------------------------------------------------------------------------

__global__ void findCellStartEndKernel(
    unsigned int* cellStart,
    unsigned int* cellEnd,
    unsigned int* gridHash,
    unsigned int* gridIndex,
    float4* position,
    float4* positionSorted,
    float4* velocity,
    float4* velocitySorted,
    int N,
    int totalCells,
    bool reorder)
{
    extern __shared__ unsigned int sharedHash[];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    unsigned int hash = 0xFFFFFFFF;
    if (idx < N) {
        hash = gridHash[idx];
        
        // Store in shared memory for comparison with neighbors
        sharedHash[threadIdx.x + 1] = hash;
        if (idx > 0 && threadIdx.x == 0) {
            sharedHash[0] = gridHash[idx - 1];
        }
    }
    
    __syncthreads();
    
    if (idx < N) {
        // Check if this is the start of a new cell
        if (idx == 0 || hash != sharedHash[threadIdx.x]) {
            cellStart[hash] = idx;
            if (idx > 0) {
                cellEnd[sharedHash[threadIdx.x]] = idx;
            }
        }
        
        // Last particle
        if (idx == N - 1) {
            cellEnd[hash] = N;
        }
        
        // Reorder position and velocity for better memory access
        if (reorder) {
            unsigned int sortedIndex = gridIndex[idx];
            positionSorted[idx] = position[sortedIndex];
            velocitySorted[idx] = velocity[sortedIndex];
        }
    }
}

//-----------------------------------------------------------------------------
// Density computation kernel
//-----------------------------------------------------------------------------

__global__ void stepDensityKernel(
    float4* position,
    float* density,
    float* pressure,
    unsigned int* gridIndex,
    unsigned int* cellStart,
    unsigned int* cellEnd,
    SPHParams params,
    bool reorder)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= params.N) return;
    
    float3 pos = make_float3_from_float4(position[idx]);
    float rho = 0.0f;
    
    int3 gridPos = getGridPosition(pos, params.numCells, params.boxsize, params.h);
    
    // Loop through neighboring cells
    for (int zi = max(0, gridPos.z - 1); zi <= min(params.numCells.z - 1, gridPos.z + 1); zi++) {
        for (int yi = max(0, gridPos.y - 1); yi <= min(params.numCells.y - 1, gridPos.y + 1); yi++) {
            for (int xi = max(0, gridPos.x - 1); xi <= min(params.numCells.x - 1, gridPos.x + 1); xi++) {
                unsigned int hash = getGridHash(make_int3(xi, yi, zi), params.numCells);
                unsigned int startIdx = cellStart[hash];
                
                if (startIdx == 0xFFFFFFFF) continue;
                
                unsigned int endIdx = cellEnd[hash];
                for (unsigned int i = startIdx; i < endIdx; i++) {
                    int j = reorder ? i : gridIndex[i];
                    float3 posJ = make_float3_from_float4(position[j]);
                    
                    float3 diff = make_float3(pos.x - posJ.x, pos.y - posJ.y, pos.z - posJ.z);
                    float dist = vlen(diff);
                    
                    rho += kernelM4(dist, params.h);
                }
            }
        }
    }
    
    rho *= params.mass;
    density[idx] = rho;
    pressure[idx] = fmaxf(0.0f, params.k * (rho - params.density0));
}

//-----------------------------------------------------------------------------
// Force computation kernel
//-----------------------------------------------------------------------------

__global__ void stepForcesKernel(
    float4* position,
    float4* velocity,
    float4* acceleration,
    float* density,
    float* pressure,
    unsigned int* gridIndex,
    unsigned int* cellStart,
    unsigned int* cellEnd,
    SPHParams params,
    bool reorder)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= params.N) return;
    
    float3 pos = make_float3_from_float4(position[idx]);
    float3 vel = make_float3_from_float4(velocity[idx]);
    float rho = density[idx];
    float p = pressure[idx];
    
    float3 accel = make_float3(0.0f, 0.0f, 0.0f);
    
    int3 gridPos = getGridPosition(pos, params.numCells, params.boxsize, params.h);
    
    // Loop through neighboring cells
    for (int zi = max(0, gridPos.z - 1); zi <= min(params.numCells.z - 1, gridPos.z + 1); zi++) {
        for (int yi = max(0, gridPos.y - 1); yi <= min(params.numCells.y - 1, gridPos.y + 1); yi++) {
            for (int xi = max(0, gridPos.x - 1); xi <= min(params.numCells.x - 1, gridPos.x + 1); xi++) {
                unsigned int hash = getGridHash(make_int3(xi, yi, zi), params.numCells);
                unsigned int startIdx = cellStart[hash];
                
                if (startIdx == 0xFFFFFFFF) continue;
                
                unsigned int endIdx = cellEnd[hash];
                for (unsigned int i = startIdx; i < endIdx; i++) {
                    int j = reorder ? i : gridIndex[i];
                    if (j == idx) continue;
                    
                    float3 posJ = make_float3_from_float4(position[j]);
                    float3 velJ = make_float3_from_float4(velocity[j]);
                    float rhoJ = density[j];
                    float pJ = pressure[j];
                    
                    float3 diff = make_float3(pos.x - posJ.x, pos.y - posJ.y, pos.z - posJ.z);
                    float dist = vlen(diff);
                    
                    if (dist > 0.0f) {
                        float3 dir = make_float3(diff.x / dist, diff.y / dist, diff.z / dist);
                        
                        // Pressure force
                        float pressureTerm = 0.5f * (p + pJ) / rhoJ * kernelM4_d(dist, params.h);
                        accel.x -= dir.x * pressureTerm;
                        accel.y -= dir.y * pressureTerm;
                        accel.z -= dir.z * pressureTerm;
                        
                        // Viscosity force
                        float viscTerm = params.viscosity / rhoJ * kernelVisc_dd(dist, params.h);
                        accel.x += (vel.x - velJ.x) * viscTerm;
                        accel.y += (vel.y - velJ.y) * viscTerm;
                        accel.z += (vel.z - velJ.z) * viscTerm;
                    }
                }
            }
        }
    }
    
    accel.x *= params.mass / rho;
    accel.y *= params.mass / rho;
    accel.z *= params.mass / rho;
    
    // Gravity
    accel.y -= 9.81f;
    
    // Store in original order if reordering was used
    int outIdx = reorder ? gridIndex[idx] : idx;
    acceleration[outIdx] = make_float4_from_float3(accel);
}

//-----------------------------------------------------------------------------
// Integration kernel
//-----------------------------------------------------------------------------

__global__ void stepMoveKernel(
    float4* position,
    float4* velocity,
    float4* acceleration,
    SPHParams params)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= params.N) return;
    
    float3 pos = make_float3_from_float4(position[idx]);
    float3 vel = make_float3_from_float4(velocity[idx]);
    float3 acc = make_float3_from_float4(acceleration[idx]);
    
    // Integrate
    vel.x += acc.x * params.dt;
    vel.y += acc.y * params.dt;
    vel.z += acc.z * params.dt;
    
    pos.x += vel.x * params.dt;
    pos.y += vel.y * params.dt;
    pos.z += vel.z * params.dt;
    
    // Boundary collision with damping
    const float damp = 0.4f;
    
    if (pos.x < 0.0f) { pos.x = 0.0f; vel.x *= -damp; }
    if (pos.y < 0.0f) { pos.y = 0.0f; vel.y *= -damp; }
    if (pos.z < 0.0f) { pos.z = 0.0f; vel.z *= -damp; }
    
    if (pos.x > params.boxsize.x) { pos.x = params.boxsize.x; vel.x *= -damp; }
    if (pos.y > params.boxsize.y) { pos.y = params.boxsize.y; vel.y *= -damp; }
    if (pos.z > params.boxsize.z) { pos.z = params.boxsize.z; vel.z *= -damp; }
    
    position[idx] = make_float4_from_float3(pos);
    velocity[idx] = make_float4_from_float3(vel);
}

//-----------------------------------------------------------------------------
// Host wrapper functions
//-----------------------------------------------------------------------------

void launchComputeHash(float4* position, unsigned int* gridHash, unsigned int* gridIndex,
                       int N, int3 numCells, float3 boxsize, float h)
{
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    computeHashKernel<<<numBlocks, BLOCK_SIZE>>>(position, gridHash, gridIndex, N, numCells, boxsize, h);
}

void launchFindCellStartEnd(unsigned int* cellStart, unsigned int* cellEnd,
                            unsigned int* gridHash, unsigned int* gridIndex,
                            float4* position, float4* positionSorted,
                            float4* velocity, float4* velocitySorted,
                            int N, int p2, int totalCells, bool reorder)
{
    // Initialize cell start to 0xFFFFFFFF
    cudaMemset(cellStart, 0xFF, totalCells * sizeof(unsigned int));
    
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t sharedMemSize = (BLOCK_SIZE + 1) * sizeof(unsigned int);
    findCellStartEndKernel<<<numBlocks, BLOCK_SIZE, sharedMemSize>>>(
        cellStart, cellEnd, gridHash, gridIndex,
        position, positionSorted, velocity, velocitySorted,
        N, totalCells, reorder);
}

void launchStepDensity(float4* position, float* density, float* pressure,
                       unsigned int* gridIndex, unsigned int* cellStart, unsigned int* cellEnd,
                       SPHParams params, bool useGrid, bool reorder)
{
    int numBlocks = (params.N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    stepDensityKernel<<<numBlocks, BLOCK_SIZE>>>(
        position, density, pressure, gridIndex, cellStart, cellEnd, params, reorder);
}

void launchStepForces(float4* position, float4* velocity, float4* acceleration,
                      float* density, float* pressure,
                      unsigned int* gridIndex, unsigned int* cellStart, unsigned int* cellEnd,
                      SPHParams params, bool useGrid, bool reorder)
{
    int numBlocks = (params.N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    stepForcesKernel<<<numBlocks, BLOCK_SIZE>>>(
        position, velocity, acceleration, density, pressure,
        gridIndex, cellStart, cellEnd, params, reorder);
}

void launchStepMove(float4* position, float4* velocity, float4* acceleration, SPHParams params)
{
    int numBlocks = (params.N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    stepMoveKernel<<<numBlocks, BLOCK_SIZE>>>(position, velocity, acceleration, params);
}

void thrustSortByKey(unsigned int* keys, unsigned int* values, int count)
{
    thrust::device_ptr<unsigned int> d_keys(keys);
    thrust::device_ptr<unsigned int> d_values(values);
    thrust::sort_by_key(d_keys, d_keys + count, d_values);
}
