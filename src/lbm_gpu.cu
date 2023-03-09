
#include <cstddef>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <sys/types.h>

#include "../include/lbm_gpu.cuh"

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

__device__ inline u_int32_t get_index(LatticeInfo &space_data, u_int32_t x,
                                      u_int32_t y, u_int32_t z) {
  return (z * space_data.x_size * space_data.y_size) + (y * space_data.x_size) +
         x;
}

__global__ void gpu_init_memory(LatticeNode *space, LatticeInfo space_data) {
  u_int32_t x = blockDim.x * blockIdx.x + threadIdx.x;
  u_int32_t y = blockDim.y * blockIdx.y + threadIdx.y;
  u_int32_t z = blockDim.z * blockIdx.z + threadIdx.z;

  if (!(x < space_data.x_size && y < space_data.y_size &&
        z < space_data.z_size))
    return;

  u_int32_t index = get_index(space_data, x, y, z);

  space[index].f[0] = index;
  /*{pos, pos, pos, pos, pos, pos, pos, pos, pos,
                pos, pos, pos, pos, pos, pos, pos, pos, pos,
                pos, pos, pos, pos, pos, pos, pos, pos, pos};*/
}

__global__ void gpu_print_memory(LatticeNode *space, LatticeInfo space_data) {
  u_int32_t x = blockDim.x * blockIdx.x + threadIdx.x;
  u_int32_t y = blockDim.y * blockIdx.y + threadIdx.y;
  u_int32_t z = blockDim.z * blockIdx.z + threadIdx.z;

  if (!(x < space_data.x_size && y < space_data.y_size &&
        z < space_data.z_size))
    return;

  u_int32_t index = get_index(space_data, x, y, z);
  printf("GPU: %i -> %i\n", index, space[index].f[0]);
}

void cuda_wait_for_device() { gpuErrchk(cudaDeviceSynchronize()); }

void lbm_space_init_device(LatticeSpace *space) {
  gpuErrchk(cudaMalloc(&space->device_data,
                       space->info.total_size * sizeof(LatticeNode)));
}

void lbm_space_init_kernel(LatticeSpace *space) {
  // TODO: these are hardcoded for 3x3x3 case
  dim3 blockSize(3, 3, 3);
  dim3 gridSize(1, 1, 1);
  gpu_init_memory<<<gridSize, blockSize>>>(space->device_data, space->info);
  gpuErrchk(cudaPeekAtLastError());
  gpu_print_memory<<<gridSize, blockSize>>>(space->device_data, space->info);
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
