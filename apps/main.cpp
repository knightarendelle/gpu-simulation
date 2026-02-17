#include "sim/sim.hpp"
#include <cuda_runtime.h>

#include <vector>
#include <random>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <iomanip>
#include <cmath>

// -------------------- CUDA error helper --------------------

static void cuda_check(cudaError_t e, const char* msg) {
  if (e != cudaSuccess) {
    std::cerr << "[CUDA ERROR] " << msg << ": "
              << cudaGetErrorString(e) << "\n";
    std::exit(1);
  }
}

// -------------------- Initial Conditions --------------------

static void init_galaxy(
  std::vector<float>& x,
  std::vector<float>& y,
  std::vector<float>& vx,
  std::vector<float>& vy,
  std::vector<float>& m
) {
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist_r(0.0f, 1.0f);
  std::uniform_real_distribution<float> dist_theta(0.0f, 2.0f * 3.1415926f);
  std::uniform_real_distribution<float> dist_mass(0.5f, 2.0f);

  const float radius = 5.0f;

  for (size_t i = 0; i < x.size(); ++i) {
    float r = std::sqrt(dist_r(rng)) * radius;
    float theta = dist_theta(rng);

    x[i] = r * std::cos(theta);
    y[i] = r * std::sin(theta);

    // Tangential velocity for orbit-like motion
    float speed = std::sqrt(1.0f / (r + 0.1f));
    vx[i] = -speed * std::sin(theta);
    vy[i] =  speed * std::cos(theta);

    m[i] = dist_mass(rng);
  }
}

// -------------------- Frame Writer --------------------

static void write_frame_csv(
  const std::string& path,
  const std::vector<float>& x,
  const std::vector<float>& y
) {
  std::ofstream f(path);
  f << "x,y\n";
  for (size_t i = 0; i < x.size(); ++i) {
    f << x[i] << "," << y[i] << "\n";
  }
}

// -------------------- MAIN --------------------

int main() {
  const std::size_t n = 30'000;      // Adjust if too slow
  const int total_steps = 600;
  const int dump_every = 2;

  sim::NBodyParams p;
  p.dt = 0.01f;
  p.G = 1.0f;
  p.softening = 0.05f;

  std::vector<float> x(n), y(n), vx(n), vy(n), m(n);

  init_galaxy(x, y, vx, vy, m);

  float *dx, *dy, *dvx, *dvy, *dm;

  cuda_check(cudaMalloc(&dx,  n * sizeof(float)), "malloc dx");
  cuda_check(cudaMalloc(&dy,  n * sizeof(float)), "malloc dy");
  cuda_check(cudaMalloc(&dvx, n * sizeof(float)), "malloc dvx");
  cuda_check(cudaMalloc(&dvy, n * sizeof(float)), "malloc dvy");
  cuda_check(cudaMalloc(&dm,  n * sizeof(float)), "malloc dm");

  cuda_check(cudaMemcpy(dx,  x.data(),  n * sizeof(float), cudaMemcpyHostToDevice), "upload x");
  cuda_check(cudaMemcpy(dy,  y.data(),  n * sizeof(float), cudaMemcpyHostToDevice), "upload y");
  cuda_check(cudaMemcpy(dvx, vx.data(), n * sizeof(float), cudaMemcpyHostToDevice), "upload vx");
  cuda_check(cudaMemcpy(dvy, vy.data(), n * sizeof(float), cudaMemcpyHostToDevice), "upload vy");
  cuda_check(cudaMemcpy(dm,  m.data(),  n * sizeof(float), cudaMemcpyHostToDevice), "upload m");

  std::filesystem::create_directories("out/frames");

  std::cout << "Running simulation...\n";

  for (int step = 0; step < total_steps; ++step) {
    sim::nbody_step_gpu(dx, dy, dvx, dvy, dm, n, p);

    if (step % dump_every == 0) {
      cuda_check(cudaDeviceSynchronize(), "sync for frame");

      cuda_check(cudaMemcpy(x.data(), dx, n * sizeof(float), cudaMemcpyDeviceToHost), "download x");
      cuda_check(cudaMemcpy(y.data(), dy, n * sizeof(float), cudaMemcpyDeviceToHost), "download y");

      std::ostringstream name;
      name << "out/frames/frame_"
           << std::setw(5) << std::setfill('0')
           << step << ".csv";

      write_frame_csv(name.str(), x, y);

      std::cout << "Wrote frame " << step << "\n";
    }
  }

  std::cout << "Simulation complete.\n";

  cudaFree(dx);
  cudaFree(dy);
  cudaFree(dvx);
  cudaFree(dvy);
  cudaFree(dm);

  return 0;
}
