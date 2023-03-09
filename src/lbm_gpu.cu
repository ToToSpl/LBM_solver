
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

__device__ inline LatticeNode *get_node_from_pitched(cudaPitchedPtr &spacePtr,
                                                     LatticeInfo &space_data,
                                                     u_int32_t x, u_int32_t y,
                                                     u_int32_t z) {
  LatticeNode *devPtr = (LatticeNode *)spacePtr.ptr;
  size_t pitch = spacePtr.pitch;
  size_t slicePitch = pitch * space_data.y_size;

  LatticeNode *elemPtr = (LatticeNode *)(devPtr + z * slicePitch + y * pitch) +
                         x * sizeof(LatticeNode);
  return elemPtr;
}

__device__ inline u_int32_t get_index(LatticeInfo &space_data, u_int32_t x,
                                      u_int32_t y, u_int32_t z) {
  return (z * space_data.x_size * space_data.y_size) + (y * space_data.x_size) +
         x;
}

__global__ void gpu_init_memory(cudaPitchedPtr spacePtr,
                                LatticeInfo space_data) {
  u_int32_t x = blockDim.x * blockIdx.x + threadIdx.x;
  u_int32_t y = blockDim.y * blockIdx.y + threadIdx.y;
  u_int32_t z = blockDim.z * blockIdx.z + threadIdx.z;

  if (!(x < space_data.x_size && y < space_data.y_size &&
        z < space_data.z_size))
    return;

  LatticeNode *elemPtr = get_node_from_pitched(spacePtr, space_data, x, y, z);
  u_int32_t index = get_index(space_data, x, y, z);

  elemPtr->f[0] = index;
  /*{pos, pos, pos, pos, pos, pos, pos, pos, pos,
                pos, pos, pos, pos, pos, pos, pos, pos, pos,
                pos, pos, pos, pos, pos, pos, pos, pos, pos};*/
}

__global__ void gpu_print_memory(cudaPitchedPtr spacePtr,
                                 LatticeInfo space_data) {
  u_int32_t x = blockDim.x * blockIdx.x + threadIdx.x;
  u_int32_t y = blockDim.y * blockIdx.y + threadIdx.y;
  u_int32_t z = blockDim.z * blockIdx.z + threadIdx.z;

  if (!(x < space_data.x_size && y < space_data.y_size &&
        z < space_data.z_size))
    return;

  LatticeNode *elemPtr = get_node_from_pitched(spacePtr, space_data, x, y, z);
  printf("%p\n", elemPtr);
  u_int32_t index = get_index(space_data, x, y, z);
  printf("GPU: %i -> %i\n", index, elemPtr->f[0]);
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
  // TODO: these are hardcoded for 3x3x3 case
  dim3 blockSize(3, 3, 3);
  dim3 gridSize(1, 1, 1);
  gpu_init_memory<<<gridSize, blockSize>>>(
      *(cudaPitchedPtr *)space->device_data, space->info);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
  gpu_print_memory<<<gridSize, blockSize>>>(
      *(cudaPitchedPtr *)space->device_data, space->info);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
}

void lbm_space_copy_host(LatticeNode **raw_data, LatticeSpace *space) {
  // cudaPitchedPtr dstPtr = *(cudaPitchedPtr *)space->device_data;
  // raw_data =
  //     (LatticeNode *)malloc(sizeof(LatticeNode) * space->info.total_size);
  // dstPtr.ptr = raw_data;
  // dstPtr.pitch = space->info.x_size * space->info.y_size *
  // sizeof(LatticeNode);
  // // std::cout << dstPtr.pitch * dstPtr.xsize * dstPtr.ysize / 4 <<
  // std::endl; cudaMemcpy3DParms cpy_params = {0}; cpy_params.dstPtr = dstPtr;
  // cpy_params.srcPtr = *(cudaPitchedPtr *)space->device_data;
  // cpy_params.extent = make_cudaExtent(sizeof(LatticeNode) *
  // space->info.x_size,
  //                                     space->info.y_size,
  //                                     space->info.z_size);
  // cpy_params.kind = cudaMemcpyDeviceToHost;
  //
  // gpuErrchk(cudaMemcpy3D(&cpy_params));
  // raw_data =
  //     (LatticeNode *)malloc(sizeof(LatticeNode) * space->info.total_size);
  // gpuErrchk(cudaMemcpy2D(raw_data, space->info.y_size * space->info.z_size,
  //                        ((cudaPitchedPtr *)space->device_data)->ptr,
  //                        ((cudaPitchedPtr *)space->device_data)->pitch,
  //                        space->info.x_size * sizeof(LatticeNode),
  //                        space->info.y_size * space->info.z_size,
  //                        cudaMemcpyDeviceToHost));

  /*
  LatticeNode *devPtr = (LatticeNode *)spacePtr.ptr;
  size_t pitch = spacePtr.pitch;
  size_t slicePitch = pitch * space_data.y_size;

  LatticeNode *elemPtr = (LatticeNode *)(devPtr + z * slicePitch + y * pitch) +
                         x * sizeof(LatticeNode);
  return elemPtr;
  */
  *raw_data =
      (LatticeNode *)malloc(sizeof(LatticeNode) * space->info.total_size);
  cudaPitchedPtr *cudaPtr = (cudaPitchedPtr *)space->device_data;
  std::cout << cudaPtr->ysize << " " << cudaPtr->xsize << " " << cudaPtr->pitch
            << std::endl;
  for (u_int32_t i = 0; i < space->info.z_size; i++) {
    for (u_int32_t j = 0; j < space->info.y_size; j++) {
      for (u_int32_t k = 0; k < space->info.x_size; k++) {
        u_int32_t index = (i * space->info.x_size * space->info.y_size) +
                          (j * space->info.x_size) + k;

        size_t slicePitch = cudaPtr->pitch * space->info.y_size;
        LatticeNode *gpu_ptr =
            (LatticeNode *)((LatticeNode *)cudaPtr->ptr + i * slicePitch +
                            j * cudaPtr->pitch) +
            k * sizeof(LatticeNode);
        printf("%p\t", gpu_ptr);
        std::cout << index << std::endl;

        gpuErrchk(cudaMemcpy(&(*raw_data)[index], gpu_ptr, sizeof(LatticeNode),
                             cudaMemcpyDeviceToHost));
        std::cout << (*raw_data)[index].f[0] << std::endl;
      }
    }
  }
}
