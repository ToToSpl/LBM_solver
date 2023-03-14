// Include C++ header files.
#include <cstddef>
#include <iostream>
#include <sys/types.h>

#include <chrono>

// Include local CUDA header files.
#include "include/lbm_constants.h"
#include "include/lbm_gpu.cuh"

int main() {

  LatticeSpace space;
  space.info.x_size = 800;
  space.info.y_size = 200;
  space.info.z_size = 300;
  space.info.total_size =
      space.info.x_size * space.info.y_size * space.info.z_size;

  LatticeCollisionType *collision = (LatticeCollisionType *)malloc(
      space.info.total_size * sizeof(LatticeCollisionType));

  for (int i = 0; i < space.info.z_size; i++) {
    for (int j = 0; j < space.info.y_size; j++) {
      for (int k = 0; k < space.info.x_size; k++) {
        size_t index = (i * space.info.x_size * space.info.y_size) +
                       (j * space.info.x_size) + k;
        collision[index] = LatticeCollisionEnum::NO_COLLISION;
      }
    }
  }

  lbm_space_init_device(&space, collision);
  lbm_space_init_kernel(&space);
  cuda_wait_for_device();

  auto start = std::chrono::high_resolution_clock::now();
  int sampling = 5;
  for (int i = 0; i < sampling; i++) {
    lbm_space_bgk_collision(&space);
    lbm_space_stream(&space);
  }
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  std::cout << (float)duration.count() / ((float)sampling * 1000.f) << "s"
            << std::endl;

  return 0;

  LatticeNode *raw_out = lbm_space_copy_host(&space);

  for (int i = 0; i < space.info.z_size; i++) {
    std::cout << "----------------" << std::endl;
    for (int j = 0; j < space.info.y_size; j++) {
      for (int k = 0; k < space.info.x_size; k++) {
        size_t index = (i * space.info.x_size * space.info.y_size) +
                       (j * space.info.x_size) + k;
        std::cout << raw_out[index].f[1] << "\t";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }

  return 0;
}
