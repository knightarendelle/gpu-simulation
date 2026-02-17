#include "sim/sim.hpp"
#include <cuda_runtime.h>

#include <vector>
#include <random>
#include <iostream>
#include <chrono>

static void init(
  std::vector<float>& x,
  std::vector<float>& y,
  std::vector<float>& vx,
  std::vector<float>& vy,
  std::vector<float>& m
) {
  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist_pos(-1.0f, 1.0f);
  std::uniform_real_distribution<float> dist_mass(0.5f, 2.0f);

  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i]  = dist_pos(rng);
    y[i]  = dist_pos(rng);
    vx[i] = 0.0f;
    vy[i] = 0.0f;
    m[i]  = dist_mass(rng);
  }
}

int main() {
  sim::NBodyParams p;
  p.dt        = 0.01f;
  p.G         = 1.0f;
  p.softening = 1e-4f;

  /* ================= CPU ================= */

  const std::size_t n_cpu = 2'000;
  const int steps_cpu = 10;

  std::vector<float> x_cpu(n_cpu), y_cpu(n_cpu),
                     vx_cpu(n_cpu), vy_cpu(n_cpu), m_cpu(n_cpu);

  init(x_cpu, y_cpu, vx_cpu, vy_cpu, m_cpu);

  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < steps_cpu; ++i)
    sim::nbody_step_cpu(
      x_cpu.data(), y_cpu.data(),
      vx_cpu.data(), vy_cpu.data(),
      m_cpu.data(), n_cpu, p
    );
  auto t1 = std::chrono::high_resolution_clock::now();

  double cpu_ms =
    std::chrono::duration<double, std::milli>(t1 - t0).count();

  std::cout << "[CPU]\n";
  std::cout << "  n = " << n_cpu << ", steps = " << steps_cpu << "\n";
  std::cout << "  time_ms = " << cpu_ms << "\n";

  /* ================= GPU ================= */

  const std::size_t n_gpu = 20'000;
  const int steps_gpu = 100;

  std::vector<float> x(n_gpu), y(n_gpu), vx(n_gpu), vy(n_gpu), m(n_gpu);
  init(x, y, vx, vy, m);

  float *dx, *dy, *dvx, *dvy, *dm;
  cudaMalloc(&dx,  n_gpu * sizeof(float));
  cudaMalloc(&dy,  n_gpu * sizeof(float));
  cudaMalloc(&dvx, n_gpu * sizeof(float));
  cudaMalloc(&dvy, n_gpu * sizeof(float));
  cudaMalloc(&dm,  n_gpu * sizeof(float));

  auto upload = [&](){
    cudaMemcpy(dx,  x.data(),  n_gpu * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dy,  y.data(),  n_gpu * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dvx, vx.data(), n_gpu * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dvy, vy.data(), n_gpu * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dm,  m.data(),  n_gpu * sizeof(float), cudaMemcpyHostToDevice);
  };

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  auto time_run = [&](const char* label, auto step_fn){
    upload();

    for (int i = 0; i < 10; ++i)
      step_fn(dx, dy, dvx, dvy, dm, n_gpu, p);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < steps_gpu; ++i)
      step_fn(dx, dy, dvx, dvy, dm, n_gpu, p);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    std::cout << label << "\n";
    std::cout << "  n = " << n_gpu << ", steps = " << steps_gpu << "\n";
    std::cout << "  time_ms = " << ms << "\n";
  };

  time_run("[GPU naive]", sim::nbody_step_gpu_naive);
  time_run("[GPU tiled]", sim::nbody_step_gpu_tiled);

  cudaFree(dx); cudaFree(dy); cudaFree(dvx); cudaFree(dvy); cudaFree(dm);
}
