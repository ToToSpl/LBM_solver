// Include C++ header files.
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

  return 0;
}
