#include "sim/sim.hpp"
#include <cuda_runtime.h>
#include <iostream>

int main() {
  const std::size_t n = 5'000'000;
  const int steps = 1000;

  float *dx, *dy, *dvx, *dvy;
  cudaMalloc(&dx,  n * sizeof(float));
  cudaMalloc(&dy,  n * sizeof(float));
  cudaMalloc(&dvx, n * sizeof(float));
  cudaMalloc(&dvy, n * sizeof(float));

  cudaMemset(dx,  0, n * sizeof(float));
  cudaMemset(dy,  0, n * sizeof(float));
  cudaMemset(dvx, 0, n * sizeof(float));
  cudaMemset(dvy, 0, n * sizeof(float));

  sim::Params p{0.016f, 0.999f};

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // warmup
  for (int i = 0; i < 50; ++i)
    sim::step_particles(dx, dy, dvx, dvy, n, p);
  cudaDeviceSynchronize();

  cudaEventRecord(start);
  for (int i = 0; i < steps; ++i)
    sim::step_particles(dx, dy, dvx, dvy, n, p);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms = 0.f;
  cudaEventElapsedTime(&ms, start, stop);

  double steps_per_sec = steps / (ms / 1000.0);
  double particles_per_sec = steps_per_sec * n;

  std::cout << "Particles: " << n << "\n";
  std::cout << "Steps: " << steps << "\n";
  std::cout << "Time (ms): " << ms << "\n";
  std::cout << "Particles/sec: " << particles_per_sec << "\n";

  cudaFree(dx);
  cudaFree(dy);
  cudaFree(dvx);
  cudaFree(dvy);
}
