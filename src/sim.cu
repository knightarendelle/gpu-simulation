#include "sim/sim.hpp"
#include <cuda_runtime.h>

namespace sim {

  __global__ void step_kernel(
    float* x,
    float* y,
    float* vx,
    float* vy,
    std::size_t n,
    Params p
  )   {
      std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i >= n) return;

      vx[i] *= p.damping;
      vy[i] *= p.damping;

      x[i] += vx[i] * p.dt;
      y[i] += vy[i] * p.dt;
  }

  void step_particles(
    float* x,
    float* y,
    float* vx,
    float* vy,
    std::size_t n,
    Params p
  ) {
    int block = 256;
    int grid  = static_cast<int>((n + block - 1) / block);
    step_kernel<<<grid, block>>>(x, y, vx, vy, n, p);
  }

}
