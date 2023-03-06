
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

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

__global__ void gpu_init_memory(LatticeNode *space, LatticeInfo space_data) {
  u_int32_t x = blockDim.x * blockIdx.x + threadIdx.x;
  u_int32_t y = blockDim.y * blockIdx.y + threadIdx.y;
  u_int32_t z = blockDim.z * blockIdx.z + threadIdx.z;

  if (x < space_data.x_size && y < space_data.y_size && z < space_data.z_size) {
    u_int32_t index = (z * space_data.x_size * space_data.y_size) +
                      (y * space_data.x_size) + x;
    float pos = (float)index;
    space[index].f[0] = 69.f;
    /*{pos, pos, pos, pos, pos, pos, pos, pos, pos,
                  pos, pos, pos, pos, pos, pos, pos, pos, pos,
                  pos, pos, pos, pos, pos, pos, pos, pos, pos};*/
  }
}

void cuda_wait_for_device() { gpuErrchk(cudaDeviceSynchronize()); }

void lbm_space_init_device(LatticeSpace *space) {
  space->device_data = malloc(sizeof(cudaPitchedPtr));
  cudaExtent volumeSizeBytes =
      make_cudaExtent(sizeof(LatticeNode) * space->info.x_size,
                      space->info.y_size, space->info.z_size);
  gpuErrchk(
      cudaMalloc3D((cudaPitchedPtr *)space->device_data, volumeSizeBytes));
}

void lbm_space_init_kernel(LatticeSpace *space) {
  dim3 blockSize(3, 3, 3);
  dim3 gridSize(1, 1, 1);
  gpu_init_memory<<<gridSize, blockSize>>>(
      (LatticeNode *)((cudaPitchedPtr *)space->device_data)->ptr, space->info);
  gpuErrchk(cudaPeekAtLastError());
}

void lbm_space_copy_host(LatticeNode *raw_data, LatticeSpace *space) {
  raw_data =
      (LatticeNode *)malloc(sizeof(LatticeNode) * space->info.total_size);
  gpuErrchk(cudaMemcpy(raw_data, ((cudaPitchedPtr *)space->device_data)->ptr,
                       sizeof(LatticeNode) * space->info.total_size,
                       cudaMemcpyDeviceToHost));
}
