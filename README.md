# GPU Simulation Lab

This repository contains a CUDA-based GPU simulation project focused on understanding and demonstrating why GPUs matter for large-scale numerical workloads.

The project is intentionally systems-oriented:

- Explicit CUDA kernels (not just high-level frameworks)
- CPU vs GPU performance comparisons
- Benchmarking and profiling as first-class concerns

---

## Goals

- Build a correct baseline GPU simulation
- Measure performance realistically (CUDA events, not wall-clock guesses)
- Incrementally optimize using CUDA-aware techniques
- Document why each optimization works

---

## Current Implementation

### N-Body Simulation

A 2D gravitational N-body simulation implemented with:

- CPU reference implementation
- Naive GPU kernel (global memory heavy)
- Shared-memory tiled kernel
- Fused integration kernel

Correctness checks are performed against the CPU implementation before benchmarking.

---

## Optimization Techniques Explored

- Shared memory tiling
- Reduction of global memory traffic
- Kernel fusion (compute + integrate)
- Block size experimentation
- Occupancy and launch statistics analysis
- Profiling via Nsight Compute

---

## Benchmarking

Benchmarks include:

- CPU vs GPU timing
- Naive vs tiled vs fused kernel comparisons
- Scaling experiments with particle count
- CUDA event–based timing for accurate device measurement

Example snapshot:

        [CPU]
        n = 2000, steps = 10
        time_ms ≈ 70 ms

        [GPU naive]
        n = 20000, steps = 100
        time_ms ≈ 260–300 ms

        [GPU tiled]
        n = 20000, steps = 100
        time_ms ≈ 240–250 ms

Performance changes are analyzed rather than assumed.

---

## Visualization

Simulation frames are exported and rendered into GIF animations using a lightweight Python pipeline.

This enables:

    - Visual sanity checks
    - Shareable demonstration artifacts
    - Presentation-ready outputs

---

## Project Structure

    include/ - Public simulation interfaces
    src/ - CUDA implementations
    apps/ - Runnable demos / simulation drivers
    bench/ - Benchmark and performance measurement code
    viz/ - Frame rendering and GIF generation
    docs/ - Design notes and performance observations

## Status

    - CUDA toolchain verified and pinned to a supported host compiler
    - N-body baseline implemented (CPU + GPU)
    - Shared-memory tiling implemented
    - Kernel fusion implemented
    - Nsight profiling completed
    - Visualization pipeline integrated

Implementation and benchmarks are added incrementally.