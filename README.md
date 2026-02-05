# GPU Simulation Lab

This repository contains a CUDA-based GPU simulation project focused on understanding
and demonstrating **why GPUs matter** for large-scale numerical workloads.

The project is intentionally systems-oriented:
- Explicit CUDA kernels (not just high-level frameworks)
- CPU vs GPU performance comparisons
- Benchmarking and profiling as first-class concerns

## Goals

- Build a correct baseline GPU simulation
- Measure performance realistically (CUDA events, not wall-clock guesses)
- Incrementally optimize using CUDA-aware techniques
- Document *why* each optimization works

## Project Structure

include/ - Public simulation interfaces
src/ - CUDA implementations
apps/ - Runnable demos / sanity checks
bench/ - Benchmark and performance measurement code
docs/ - Notes on design and performance findings

## Status

Initial repository structure created.  
CUDA toolchain verified and pinned to a supported host compiler.

Implementation and benchmarks are added incrementally.