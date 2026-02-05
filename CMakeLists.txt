cmake_minimum_required(VERSION 3.24)
project(gpu_sim LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# RTX 4050 = Ada Lovelace
set(CMAKE_CUDA_ARCHITECTURES 89)

add_library(sim STATIC
  src/sim.cu
)

target_include_directories(sim PUBLIC include)
set_target_properties(sim PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(sim_run apps/main.cpp)
target_link_libraries(sim_run PRIVATE sim)

add_executable(sim_bench bench/bench.cpp)
target_link_libraries(sim_bench PRIVATE sim)
