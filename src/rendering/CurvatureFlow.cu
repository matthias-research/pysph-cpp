/**
 * Curvature Flow CUDA Kernel - Port of Python pysph curvature_flow.cl
 * 
 * Implements mean curvature flow to smooth the depth buffer,
 * removing the "blobby" appearance of individual particles.
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <surface_functions.h>

// Block dimensions for 2D kernel
#define BLOCK_WIDTH 16
#define BLOCK_HEIGHT 16

//-----------------------------------------------------------------------------
// Device helper functions
//-----------------------------------------------------------------------------

__device__ inline float readDepth(cudaSurfaceObject_t surf, int x, int y, int width, int height) {
    if (x < 0 || x >= width || y < 0 || y >= height) return 0.0f;
    float val;
    surf2Dread(&val, surf, x * sizeof(float), y);
    return val;
}

__device__ inline float diffZ(cudaSurfaceObject_t surf, int x, int y, int ox, int oy, int width, int height) {
    float dp = readDepth(surf, x + ox, y + oy, width, height);
    float dm = readDepth(surf, x - ox, y - oy, width, height);
    if (dp == 0.0f || dm == 0.0f) return 0.0f;
    return (dp - dm) * 0.5f;
}

__device__ inline float diffZ_2(cudaSurfaceObject_t surf, int x, int y, int ox, int oy, int width, int height) {
    float dp = readDepth(surf, x + ox, y + oy, width, height);
    float d = readDepth(surf, x, y, width, height);
    float dm = readDepth(surf, x - ox, y - oy, width, height);
    return dp - 2.0f * d + dm;
}

__device__ inline float diffZ_xy(cudaSurfaceObject_t surf, int x, int y, int width, int height) {
    float pp = readDepth(surf, x + 1, y + 1, width, height);
    float pm = readDepth(surf, x + 1, y - 1, width, height);
    float mp = readDepth(surf, x - 1, y + 1, width, height);
    float mm = readDepth(surf, x - 1, y - 1, width, height);
    return (pp - pm - mp + mm) * 0.25f;
}

// Compute mean curvature (divergence of normal)
__device__ inline float computeMeanCurvature(cudaSurfaceObject_t surf, int x, int y,
                                              int width, int height, float projW, float projH) {
    float z = readDepth(surf, x, y, width, height);
    float z_x = diffZ(surf, x, y, 1, 0, width, height);
    float z_y = diffZ(surf, x, y, 0, 1, width, height);
    
    float Cx = -2.0f / (width * projW);
    float Cy = -2.0f / (height * projH);
    
    float Wx = (width - 2.0f * x) / (width * projW);
    float Wy = (height - 2.0f * y) / (height * projH);
    
    float D = Cy * Cy * z_x * z_x + Cx * Cx * z_y * z_y + Cx * Cx * Cy * Cy * z * z;
    
    float z_xx = diffZ_2(surf, x, y, 1, 0, width, height);
    float z_yy = diffZ_2(surf, x, y, 0, 1, width, height);
    float z_xy = diffZ_xy(surf, x, y, width, height);
    
    float D_x = 2.0f * Cy * Cy * z_x * z_xx + 2.0f * Cx * Cx * z_y * z_xy + 2.0f * Cx * Cx * Cy * Cy * z * z_x;
    float D_y = 2.0f * Cy * Cy * z_x * z_xy + 2.0f * Cx * Cx * z_y * z_yy + 2.0f * Cx * Cx * Cy * Cy * z * z_y;
    
    float Ex = 0.5f * z_x * D_x - z_xx * D;
    float Ey = 0.5f * z_y * D_y - z_yy * D;
    
    float H = (Cy * Ex + Cx * Ey) / (2.0f * D * sqrtf(D));
    return H;
}

//-----------------------------------------------------------------------------
// Curvature flow kernel
//-----------------------------------------------------------------------------

__global__ void curvatureFlowKernel(
    cudaSurfaceObject_t depthIn,
    cudaSurfaceObject_t depthOut,
    int width, int height,
    float dt, float zContrib,
    float projW, float projH)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    float depth = readDepth(depthIn, x, y, width, height);
    
    if (depth == 0.0f) {
        surf2Dwrite(depth, depthOut, x * sizeof(float), y);
        return;
    }
    
    float z_x = diffZ(depthIn, x, y, 1, 0, width, height);
    float z_y = diffZ(depthIn, x, y, 0, 1, width, height);
    
    float meanCurv = computeMeanCurvature(depthIn, x, y, width, height, projW, projH);
    
    // Adaptive smoothing: more smoothing where depth varies a lot
    depth += meanCurv * dt * (1.0f + (fabsf(z_x) + fabsf(z_y)) * zContrib);
    
    surf2Dwrite(depth, depthOut, x * sizeof(float), y);
}

//-----------------------------------------------------------------------------
// Host wrapper function
//-----------------------------------------------------------------------------

void launchCurvatureFlow(cudaSurfaceObject_t depthIn, cudaSurfaceObject_t depthOut,
                         int width, int height, float dt, float zContrib,
                         float projW, float projH)
{
    dim3 blockDim(BLOCK_WIDTH, BLOCK_HEIGHT);
    dim3 gridDim((width + BLOCK_WIDTH - 1) / BLOCK_WIDTH,
                 (height + BLOCK_HEIGHT - 1) / BLOCK_HEIGHT);
    
    curvatureFlowKernel<<<gridDim, blockDim>>>(depthIn, depthOut, width, height,
                                                dt, zContrib, projW, projH);
    cudaDeviceSynchronize();
}
