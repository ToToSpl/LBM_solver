
#include <cstddef>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <sys/types.h>

#include "../include/lbm_constants.h"
#include "../include/lbm_gpu.cuh"
#include "./lbm_helpers.cuh"

// ---- KERNELS -------

__device__ inline size_t get_index(LatticeInfo &space_data, u_int32_t x,
                                   u_int32_t y, u_int32_t z) {
  return (z * space_data.x_size * space_data.y_size) + (y * space_data.x_size) +
         x;
}

// Standard procedure, check if core is in space and initialize x,y,z and index
#define KERNEL_ONE_ELEMENT_INIT                                                \
  u_int32_t x = blockDim.x * blockIdx.x + threadIdx.x;                         \
  u_int32_t y = blockDim.y * blockIdx.y + threadIdx.y;                         \
  u_int32_t z = blockDim.z * blockIdx.z + threadIdx.z;                         \
  if (!(x < space_data.x_size && y < space_data.y_size &&                      \
        z < space_data.z_size))                                                \
    return;                                                                    \
  size_t index = get_index(space_data, x, y, z);

__global__ void gpu_init_memory(LatticeNode *space, LatticeInfo space_data) {
  KERNEL_ONE_ELEMENT_INIT

  // set ones in each speed
  for (int i = 0; i < LBM_SPEED_COUNTS; i++)
    space[index].f[i] = 1.0f;
}

__global__ void gpu_collision_bgk(LatticeNode *space, LatticeInfo space_data) {
  KERNEL_ONE_ELEMENT_INIT
  LatticeNode node = space[index];

  Vec3 spd_vecs[] = LBM_SPEED_VECTORS;
  float spd_weights[] = LBM_SPEED_WEIGHTS;

  float rho = 0.f;
  Vec3 u = {0.f, 0.f, 0.f};
  for (u_int8_t i = 0; i < LBM_SPEED_COUNTS; i++) {
    rho += node.f[i];
    u.x += spd_vecs[i].x * node.f[i];
    u.y += spd_vecs[i].y * node.f[i];
    u.z += spd_vecs[i].z * node.f[i];
  }
  u.x /= rho;
  u.y /= rho;
  u.z /= rho;

  float elem3 = u.x * u.x + u.y * u.y + u.z * u.z;
  for (u_int8_t i = 0; i < LBM_SPEED_COUNTS; i++) {
    Vec3 spd = spd_vecs[i];
    float elem1 = u.x * spd.x + u.y * spd.y + u.z * spd.z;
    float elem2 = elem1 * elem1;

    float f_eq =
        spd_weights[i] * rho *
        (1.f + elem1 / CS2 + elem2 / (2 * CS2 * CS2) - elem3 / (2 * CS2));

    float omega = -(node.f[i] - f_eq) * SIMULATION_DT_TAU;

    node.f[i] += omega;
  }
}

// ---- END KERNELS -------

#define gpuErrchk(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

void cuda_wait_for_device() { gpuErrchk(cudaDeviceSynchronize()); }

void lbm_space_init_device(LatticeSpace *space) {
  gpuErrchk(cudaMalloc(&space->device_data,
                       space->info.total_size * sizeof(LatticeNode)));

  gpuErrchk(cudaMalloc(&space->device_output,
                       space->info.total_size * sizeof(LatticeOutput)));
}

void lbm_space_init_kernel(LatticeSpace *space) {
  ComputeDim compute_dim = compute_dim_create(
      space->info.x_size, space->info.y_size, space->info.z_size);

  gpu_init_memory<<<compute_dim.gridSize, compute_dim.blockSize>>>(
      space->device_data, space->info);
  gpuErrchk(cudaPeekAtLastError());
}

LatticeNode *lbm_space_copy_host(LatticeSpace *space) {
  LatticeNode *raw_data =
      (LatticeNode *)malloc(sizeof(LatticeNode) * space->info.total_size);
  gpuErrchk(cudaMemcpy(raw_data, space->device_data,
                       space->info.total_size * sizeof(LatticeNode),
                       cudaMemcpyDeviceToHost));
  return raw_data;
}
