// Include C++ header files.
#include <cstddef>
#include <iostream>
#include <sys/types.h>

// Include local CUDA header files.
#include "include/cuda_kernel.cuh"
#include "include/lbm_gpu.cuh"

int main() {

  LatticeSpace space;
  space.info.x_size = 3;
  space.info.y_size = 3;
  space.info.z_size = 3;
  space.info.total_size =
      space.info.x_size * space.info.y_size * space.info.z_size;

  lbm_space_init_device(&space);
  lbm_space_init_kernel(&space);
  cuda_wait_for_device();

  LatticeNode *raw_out = lbm_space_copy_host(&space);

  for (int i = 0; i < space.info.z_size; i++) {
    std::cout << "\t-----------------------" << std::endl;
    for (int j = 0; j < space.info.y_size; j++) {
      std::cout << "\t";
      for (int k = 0; k < space.info.x_size; k++) {
        size_t index = (i * space.info.x_size * space.info.y_size) +
                       (j * space.info.x_size) + k;
        std::cout << raw_out[index].f[0] << "\t";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }

  return 0;
}
