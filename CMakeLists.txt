cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(lbm_solver LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

find_package(CUDA)
find_package(CUDAToolkit)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

include_directories("${CUDA_INCLUDE_DIRS}")

set(CMAKE_CXX_FLAGS_RELEASE "-O3")

add_executable(lbm_solver 
  lbm_solver.cpp
  src/data_compressor.cpp
  src/lbm_gpu.cu
  src/lbm_helpers.cu
  )

target_compile_features(lbm_solver PUBLIC cxx_std_17)

set_target_properties( lbm_solver
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                      )

target_link_libraries(lbm_solver PRIVATE CUDA::cudart zip)


add_executable(lbm_post_processor 
  lbm_post_processor.cpp
  )

target_compile_features(lbm_post_processor  PUBLIC cxx_std_17)

set_target_properties( lbm_post_processor
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                      )

target_link_libraries(lbm_post_processor PRIVATE CUDA::cudart zip)
