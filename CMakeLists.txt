cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(lbm_solver LANGUAGES CXX CUDA)

include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

# set(CMAKE_CXX_FLAGS_RELEASE "-O3")

add_executable(lbm_solver 
  main.cpp
  src/cuda_kernel.cu
  )

target_compile_features(lbm_solver PUBLIC cxx_std_17)

set_target_properties( lbm_solver
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                      )

target_link_libraries(lbm_solver)
