// Include C++ header files.
#include <cstddef>
#include <iostream>
#include <sys/types.h>

// Include local CUDA header files.
#include "include/lbm_gpu.cuh"

int main() {

  LatticeSpace space;
  space.info.x_size = 100;
  space.info.y_size = 100;
  space.info.z_size = 100;
  space.info.total_size =
      space.info.x_size * space.info.y_size * space.info.z_size;

  lbm_space_init_device(&space);
  lbm_space_init_kernel(&space);
  cuda_wait_for_device();

  LatticeNode *raw_out = lbm_space_copy_host(&space);

  for (int i = 0; i < space.info.z_size; i++) {
    for (int j = 0; j < space.info.y_size; j++) {
      for (int k = 0; k < space.info.x_size; k++) {
        size_t index = (i * space.info.x_size * space.info.y_size) +
                       (j * space.info.x_size) + k;
        if (index != raw_out[index].f[0]) {
          std::cout << "Failed comparison at: " << index << std::endl;
          return 0;
        }
      }
    }
  }
  std::cout << "Comparison passed" << std::endl;

  return 0;
}
