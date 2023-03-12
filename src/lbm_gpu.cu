
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

__device__ inline size_t get_index(LatticeInfo &space_data, size_t x, size_t y,
                                   size_t z) {
  return (z * space_data.x_size * space_data.y_size) + (y * space_data.x_size) +
         x;
}

// Standard procedure, check if core is in space and initialize x,y,z and index
#define KERNEL_ONE_ELEMENT_INIT                                                \
  size_t x = blockDim.x * blockIdx.x + threadIdx.x;                            \
  size_t y = blockDim.y * blockIdx.y + threadIdx.y;                            \
  size_t z = blockDim.z * blockIdx.z + threadIdx.z;                            \
  if (x >= space_data.x_size || y >= space_data.y_size ||                      \
      z >= space_data.z_size)                                                  \
    return;                                                                    \
  size_t index = get_index(space_data, x, y, z);

__global__ void gpu_init_memory(LatticeNode *space, LatticeInfo space_data) {
  KERNEL_ONE_ELEMENT_INIT

  // set ones in each speed
  for (int i = 0; i < LBM_SPEED_COUNTS; i++)
    space[index].f[i] = (float)index; // 1.0f;
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

__global__ void gpu_stream_x_plus(LatticeNode *space, LatticeInfo space_data) {
  size_t y = blockDim.y * blockIdx.y + threadIdx.y;
  size_t z = blockDim.z * blockIdx.z + threadIdx.z;
  if (y >= space_data.y_size || z >= space_data.z_size)
    return;

  u_int8_t x_plus[] = LBM_STREAM_X_PLUS;
  float x_moved[sizeof(x_plus) / sizeof(x_plus[0])];

  // padding is the distance between indecies. In linear memory this is correct
  size_t index = get_index(space_data, 0, y, z);
  size_t padding = get_index(space_data, 1, y, z) - index;
  // special case 0, we replace last row
  {
    size_t last_index = get_index(space_data, space_data.x_size - 1, y, z);
    for (u_int8_t i = 0; i < (sizeof(x_plus) / sizeof(x_plus[0])); i++) {
      x_moved[i] = space[index].f[x_plus[i]];
      space[index].f[x_plus[i]] = space[last_index].f[x_plus[i]];
    }
  }

  for (size_t x = 1; x < space_data.x_size; x++) {
    index += padding;
    for (u_int8_t i = 0; i < (sizeof(x_plus) / sizeof(x_plus[0])); i++) {
      float temp = space[index].f[x_plus[i]];
      space[index].f[x_plus[i]] = x_moved[i];
      x_moved[i] = temp;
    }
  }
}

__global__ void gpu_stream_x_minus(LatticeNode *space, LatticeInfo space_data) {
  size_t y = blockDim.y * blockIdx.y + threadIdx.y;
  size_t z = blockDim.z * blockIdx.z + threadIdx.z;
  if (y >= space_data.y_size || z >= space_data.z_size)
    return;

  u_int8_t x_minus[] = LBM_STREAM_X_MINUS;
  float x_moved[sizeof(x_minus) / sizeof(x_minus[0])];

  // padding is the distance between indecies. In linear memory this is correct
  size_t index = get_index(space_data, space_data.x_size - 1, y, z);
  size_t padding = index - get_index(space_data, space_data.x_size - 2, y, z);
  // special case last, we replace first row
  {
    size_t first_index = get_index(space_data, 0, y, z);
    for (u_int8_t i = 0; i < (sizeof(x_minus) / sizeof(x_minus[0])); i++) {
      x_moved[i] = space[index].f[x_minus[i]];
      space[index].f[x_minus[i]] = space[first_index].f[x_minus[i]];
    }
  }

  for (size_t x = 1; x < space_data.x_size; x++) {
    index -= padding;
    for (u_int8_t i = 0; i < (sizeof(x_minus) / sizeof(x_minus[0])); i++) {
      float temp = space[index].f[x_minus[i]];
      space[index].f[x_minus[i]] = x_moved[i];
      x_moved[i] = temp;
    }
  }
}

__global__ void gpu_stream_y_plus(LatticeNode *space, LatticeInfo space_data) {
  size_t x = blockDim.x * blockIdx.x + threadIdx.x;
  size_t z = blockDim.z * blockIdx.z + threadIdx.z;
  if (x >= space_data.x_size || z >= space_data.z_size)
    return;

  u_int8_t y_plus[] = LBM_STREAM_Y_PLUS;
  float y_moved[sizeof(y_plus) / sizeof(y_plus[0])];

  // padding is the distance between indecies. In linear memory this is correct
  size_t index = get_index(space_data, x, 0, z);
  size_t padding = get_index(space_data, x, 1, z) - index;
  // special case 0, we replace last row
  {
    size_t last_index = get_index(space_data, x, space_data.y_size - 1, z);
    for (u_int8_t i = 0; i < (sizeof(y_plus) / sizeof(y_plus[0])); i++) {
      y_moved[i] = space[index].f[y_plus[i]];
      space[index].f[y_plus[i]] = space[last_index].f[y_plus[i]];
    }
  }

  for (size_t y = 1; y < space_data.y_size; y++) {
    index += padding;
    for (u_int8_t i = 0; i < (sizeof(y_plus) / sizeof(y_plus[0])); i++) {
      float temp = space[index].f[y_plus[i]];
      space[index].f[y_plus[i]] = y_moved[i];
      y_moved[i] = temp;
    }
  }
}

__global__ void gpu_stream_y_minus(LatticeNode *space, LatticeInfo space_data) {
  size_t x = blockDim.x * blockIdx.x + threadIdx.x;
  size_t z = blockDim.z * blockIdx.z + threadIdx.z;
  if (x >= space_data.x_size || z >= space_data.z_size)
    return;

  u_int8_t y_minus[] = LBM_STREAM_Y_MINUS;
  float y_moved[sizeof(y_minus) / sizeof(y_minus[0])];

  // padding is the distance between indecies. In linear memory this is correct
  size_t index = get_index(space_data, x, space_data.y_size - 1, z);
  size_t padding = index - get_index(space_data, x, space_data.y_size - 2, z);
  // special case last, we replace first row
  {
    size_t first_index = get_index(space_data, x, 0, z);
    for (u_int8_t i = 0; i < (sizeof(y_minus) / sizeof(y_minus[0])); i++) {
      y_moved[i] = space[index].f[y_minus[i]];
      space[index].f[y_minus[i]] = space[first_index].f[y_minus[i]];
    }
  }

  for (size_t y = 1; y < space_data.y_size; y++) {
    index -= padding;
    for (u_int8_t i = 0; i < (sizeof(y_minus) / sizeof(y_minus[0])); i++) {
      float temp = space[index].f[y_minus[i]];
      space[index].f[y_minus[i]] = y_moved[i];
      y_moved[i] = temp;
    }
  }
}

__global__ void gpu_stream_z_plus(LatticeNode *space, LatticeInfo space_data) {
  size_t x = blockDim.x * blockIdx.x + threadIdx.x;
  size_t y = blockDim.y * blockIdx.y + threadIdx.y;
  if (x >= space_data.x_size || y >= space_data.y_size)
    return;

  u_int8_t z_plus[] = LBM_STREAM_Z_PLUS;
  float z_moved[sizeof(z_plus) / sizeof(z_plus[0])];

  // padding is the distance between indecies. In linear memory this is correct
  size_t index = get_index(space_data, x, y, 0);
  size_t padding = get_index(space_data, x, y, 1) - index;
  // special case 0, we replace last row
  {
    size_t last_index = get_index(space_data, x, y, space_data.z_size - 1);
    for (u_int8_t i = 0; i < (sizeof(z_plus) / sizeof(z_plus[0])); i++) {
      z_moved[i] = space[index].f[z_plus[i]];
      space[index].f[z_plus[i]] = space[last_index].f[z_plus[i]];
    }
  }

  for (size_t z = 1; z < space_data.z_size; z++) {
    index += padding;
    for (u_int8_t i = 0; i < (sizeof(z_plus) / sizeof(z_plus[0])); i++) {
      float temp = space[index].f[z_plus[i]];
      space[index].f[z_plus[i]] = z_moved[i];
      z_moved[i] = temp;
    }
  }
}

__global__ void gpu_stream_z_minus(LatticeNode *space, LatticeInfo space_data) {
  size_t x = blockDim.x * blockIdx.x + threadIdx.x;
  size_t y = blockDim.y * blockIdx.y + threadIdx.y;
  if (x >= space_data.x_size || y >= space_data.y_size)
    return;

  u_int8_t z_minus[] = LBM_STREAM_Z_MINUS;
  float z_moved[sizeof(z_minus) / sizeof(z_minus[0])];

  // padding is the distance between indecies. In linear memory this is correct
  size_t index = get_index(space_data, x, y, space_data.z_size - 1);
  size_t padding = index - get_index(space_data, x, y, space_data.z_size - 2);
  // special case last, we replace first row
  {
    size_t first_index = get_index(space_data, x, y, 0);
    for (u_int8_t i = 0; i < (sizeof(z_minus) / sizeof(z_minus[0])); i++) {
      z_moved[i] = space[index].f[z_minus[i]];
      space[index].f[z_minus[i]] = space[first_index].f[z_minus[i]];
    }
  }

  for (size_t z = 1; z < space_data.z_size; z++) {
    index -= padding;
    for (u_int8_t i = 0; i < (sizeof(z_minus) / sizeof(z_minus[0])); i++) {
      float temp = space[index].f[z_minus[i]];
      space[index].f[z_minus[i]] = z_moved[i];
      z_moved[i] = temp;
    }
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

void lbm_space_bgk_collision(LatticeSpace *space) {
  ComputeDim compute_dim = compute_dim_create(
      space->info.x_size, space->info.y_size, space->info.z_size);

  gpu_collision_bgk<<<compute_dim.gridSize, compute_dim.blockSize>>>(
      space->device_data, space->info);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
}

void lbm_space_stream(LatticeSpace *space) {
  ComputeDim plane_xy =
      compute_dim_create(space->info.x_size, space->info.y_size, 1);
  ComputeDim plane_xz =
      compute_dim_create(space->info.x_size, 1, space->info.z_size);
  ComputeDim plane_yz =
      compute_dim_create(1, space->info.y_size, space->info.z_size);

  gpu_stream_x_plus<<<plane_yz.gridSize, plane_yz.blockSize>>>(
      space->device_data, space->info);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  gpu_stream_x_minus<<<plane_yz.gridSize, plane_yz.blockSize>>>(
      space->device_data, space->info);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  gpu_stream_y_plus<<<plane_xz.gridSize, plane_xz.blockSize>>>(
      space->device_data, space->info);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  gpu_stream_y_minus<<<plane_xz.gridSize, plane_xz.blockSize>>>(
      space->device_data, space->info);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  gpu_stream_z_plus<<<plane_xy.gridSize, plane_xy.blockSize>>>(
      space->device_data, space->info);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  gpu_stream_z_minus<<<plane_xy.gridSize, plane_xy.blockSize>>>(
      space->device_data, space->info);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
}

LatticeNode *lbm_space_copy_host(LatticeSpace *space) {
  LatticeNode *raw_data =
      (LatticeNode *)malloc(sizeof(LatticeNode) * space->info.total_size);
  gpuErrchk(cudaMemcpy(raw_data, space->device_data,
                       space->info.total_size * sizeof(LatticeNode),
                       cudaMemcpyDeviceToHost));
  return raw_data;
}
