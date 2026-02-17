#include "sim/sim.hpp"
#include <cmath>

namespace sim {

    void nbody_step_cpu(
    float* x, float* y,
    float* vx, float* vy,
    const float* m,
    std::size_t n,
    NBodyParams p
    ) {
    for (std::size_t i = 0; i < n; ++i) {
        float ax = 0.0f, ay = 0.0f;

        const float xi = x[i];
        const float yi = y[i];

        for (std::size_t j = 0; j < n; ++j) {
        if (j == i) continue;

        float dx = x[j] - xi;
        float dy = y[j] - yi;

        float r2 = dx*dx + dy*dy + p.softening*p.softening;
        float invR = 1.0f / std::sqrt(r2);
        float invR3 = invR * invR * invR;

        float s = p.G * m[j] * invR3;
        ax += dx * s;
        ay += dy * s;
        }

        vx[i] += ax * p.dt;
        vy[i] += ay * p.dt;
    }

    for (std::size_t i = 0; i < n; ++i) {
        x[i] += vx[i] * p.dt;
        y[i] += vy[i] * p.dt;
    }
    }

} // namespace sim
