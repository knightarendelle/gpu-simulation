#include "sim/sim.hpp"
#include <cuda_runtime.h>

namespace sim {

    static constexpr int BLOCK = 128;

    // ------------------- Naive kernels -------------------

    __global__ void nbody_accel_naive(
    float* x, float* y,
    float* vx, float* vy,
    const float* m,
    std::size_t n,
    NBodyParams p
    ) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float xi = x[i];
    float yi = y[i];

    float ax = 0.0f, ay = 0.0f;

    for (std::size_t j = 0; j < n; ++j) {
        if (j == i) continue;

        float dx = x[j] - xi;
        float dy = y[j] - yi;

        float r2 = dx*dx + dy*dy + p.softening*p.softening;
        float invR = rsqrtf(r2);
        float invR3 = invR * invR * invR;

        float s = p.G * m[j] * invR3;
        ax += dx * s;
        ay += dy * s;
    }

    vx[i] += ax * p.dt;
    vy[i] += ay * p.dt;
    }

    // ------------------- Tiled Fused kernel -------------------

    __global__ void nbody_tiled_fused(
        float* x, float* y,
        float* vx, float* vy,
        const float* m,
        std::size_t n,
        NBodyParams p
      ) {
        std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;
      
        float xi  = x[i];
        float yi  = y[i];
        float vxi = vx[i];
        float vyi = vy[i];
      
        float ax = 0.0f, ay = 0.0f;
      
        __shared__ float sx[BLOCK];
        __shared__ float sy[BLOCK];
        __shared__ float sm[BLOCK];
      
        std::size_t tiles = (n + BLOCK - 1) / BLOCK;
      
        for (std::size_t t = 0; t < tiles; ++t) {
          std::size_t j = t * BLOCK + threadIdx.x;
      
          if (j < n) {
            sx[threadIdx.x] = x[j];
            sy[threadIdx.x] = y[j];
            sm[threadIdx.x] = m[j];
          } else {
            sx[threadIdx.x] = 0.0f;
            sy[threadIdx.x] = 0.0f;
            sm[threadIdx.x] = 0.0f;
          }
      
          __syncthreads();
      
          for (int k = 0; k < BLOCK; ++k) {
            std::size_t jj = t * BLOCK + (std::size_t)k;
            if (jj == i || jj >= n) continue;
      
            float dx = sx[k] - xi;
            float dy = sy[k] - yi;
      
            float r2 = dx*dx + dy*dy + p.softening*p.softening;
            float invR = rsqrtf(r2);
            float invR3 = invR * invR * invR;
      
            float s = p.G * sm[k] * invR3;
            ax += dx * s;
            ay += dy * s;
          }
      
          __syncthreads();
        }
      
        // Fused integrate
        vxi += ax * p.dt;
        vyi += ay * p.dt;
        xi  += vxi * p.dt;
        yi  += vyi * p.dt;
      
        vx[i] = vxi;
        vy[i] = vyi;
        x[i]  = xi;
        y[i]  = yi;
      }      

    // ------------------- Integrate kernel -------------------

    __global__ void nbody_integrate(
    float* x, float* y,
    const float* vx, const float* vy,
    std::size_t n,
    float dt
    ) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] += vx[i] * dt;
    y[i] += vy[i] * dt;
    }

    // ------------------- Host wrappers -------------------

    void nbody_step_gpu_naive(
    float* x, float* y,
    float* vx, float* vy,
    const float* m,
    std::size_t n,
    NBodyParams p
    ) {
    int grid = (int)((n + BLOCK - 1) / BLOCK);
    nbody_accel_naive<<<grid, BLOCK>>>(x, y, vx, vy, m, n, p);
    nbody_integrate<<<grid, BLOCK>>>(x, y, vx, vy, n, p.dt);
    }

    void nbody_step_gpu_tiled(
        float* x, float* y,
        float* vx, float* vy,
        const float* m,
        std::size_t n,
        NBodyParams p
      ) {
        int grid = (int)((n + BLOCK - 1) / BLOCK);
        nbody_tiled_fused<<<grid, BLOCK>>>(x, y, vx, vy, m, n, p);
    }

    // Default GPU step: use tiled (the fast one)
    void nbody_step_gpu(
    float* x, float* y,
    float* vx, float* vy,
    const float* m,
    std::size_t n,
    NBodyParams p
    ) {
    nbody_step_gpu_tiled(x, y, vx, vy, m, n, p);
    }

} // namespace sim
