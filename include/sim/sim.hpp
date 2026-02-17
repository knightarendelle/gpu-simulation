#pragma once
#include <cstddef>

namespace sim {

  struct NBodyParams {
    float dt;          // timestep
    float G;           // gravitational constant (scaled)
    float softening;   // epsilon to avoid singularities
  };

  void nbody_step_cpu(
    float* x, float* y,
    float* vx, float* vy,
    const float* m,
    std::size_t n,
    NBodyParams p
  );

  void nbody_step_gpu(
    float* x, float* y,
    float* vx, float* vy,
    const float* m,
    std::size_t n,
    NBodyParams p
  );

  void nbody_step_gpu_naive(
    float* x, float* y,
    float* vx, float* vy,
    const float* m,
    std::size_t n,
    NBodyParams p
  );
  
  void nbody_step_gpu_tiled(
    float* x, float* y,
    float* vx, float* vy,
    const float* m,
    std::size_t n,
    NBodyParams p
  );

} // namespace sim
