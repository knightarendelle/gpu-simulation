#include "sim/sim.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

int main() {
  const std::size_t n = 1'000'000;

  std::vector<float> hx(n, 0.f), hy(n, 0.f);
  std::vector<float> hvx(n, 1.f), hvy(n, 1.f);

  float *dx, *dy, *dvx, *dvy;
  cudaMalloc(&dx,  n * sizeof(float));
  cudaMalloc(&dy,  n * sizeof(float));
  cudaMalloc(&dvx, n * sizeof(float));
  cudaMalloc(&dvy, n * sizeof(float));

  cudaMemcpy(dx,  hx.data(),  n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dy,  hy.data(),  n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dvx, hvx.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dvy, hvy.data(), n * sizeof(float), cudaMemcpyHostToDevice);

  sim::Params p{0.016f, 0.999f};

  for (int i = 0; i < 600; ++i)
    sim::step_particles(dx, dy, dvx, dvy, n, p);

  cudaDeviceSynchronize();

  cudaMemcpy(hx.data(), dx, n * sizeof(float), cudaMemcpyDeviceToHost);
  std::cout << "x[0] = " << hx[0] << "\n";

  cudaFree(dx);
  cudaFree(dy);
  cudaFree(dvx);
  cudaFree(dvy);
}
