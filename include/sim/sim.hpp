#pragma once
#include <cstddef>

namespace sim {

struct Params {
  float dt;
  float damping;
};

void step_particles(
  float* x,
  float* y,
  float* vx,
  float* vy,
  std::size_t n,
  Params p
);

}
