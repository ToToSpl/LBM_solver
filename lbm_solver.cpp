// Include C++ header files.
#include <chrono>
#include <cstddef>
#include <iostream>
#include <string>
#include <sys/types.h>

// Include local CUDA header files.
#include "include/lbm_constants.h"
#include "include/lbm_gpu.cuh"
#include "include/lbm_types.h"
#include "src/data_compressor.hpp"

LatticeSpace create_cylinder_experiment() {
  size_t width = 400, height = 100, depth = 100;
  size_t cyl_x = width / 4, cyl_y = height / 2,
         cyl_r2 = (height / 4) * (height / 4);
  LatticeSpace space;
  space.info.x_size = width;
  space.info.y_size = height;
  space.info.z_size = depth;
  space.info.total_size = width * height * depth;

  LatticeCollision *collision = (LatticeCollision *)malloc(
      space.info.total_size * sizeof(LatticeCollision));

  // define inlet and outlet speed
  Vec3 u_in = {-0.1f, 0.f, 0.f};
  space.info.wall_speeds.s1 = {u_in, InletDir::X_PLUS};  // inlet
  space.info.wall_speeds.s2 = {u_in, InletDir::X_MINUS}; // outlet

  for (int z = 0; z < depth; z++) {
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        size_t index = (z * width * height) + (y * width) + x;
        collision[index] = LatticeCollisionEnum::NO_COLLISION;

        // // side walls
        // if (y == 0 || y == height - 1 || z == 0 || z == depth - 1)
        //   collision[index] = LatticeCollisionEnum::BOUNCE_BACK_STATIC;
        // // inlet
        // else if (x == 0)
        //   collision[index] = LatticeCollisionEnum::BOUNCE_BACK_SPEED_1;
        // // outlet
        // else if (x == width - 1)
        //   collision[index] = LatticeCollisionEnum::BOUNCE_BACK_SPEED_2;

        // // cylinder
        // if ((x - cyl_x) * (x - cyl_x) + (y - cyl_y) * (y - cyl_y) < cyl_r2)
        //   collision[index] = LatticeCollisionEnum::BOUNCE_BACK_STATIC;
        // // normal space
        // else
        //   collision[index] = LatticeCollisionEnum::NO_COLLISION;
      }
    }
  }

  lbm_space_init_device(&space, collision);
  free(collision);

  lbm_space_init_kernel(&space, 1.0f);
  cuda_wait_for_device();
  return space;
}

LatticeSpace create_test_experiment(int one_index) {
  size_t width = 7, height = 7, depth = 7;
  LatticeSpace space;
  space.info.x_size = width;
  space.info.y_size = height;
  space.info.z_size = depth;
  space.info.total_size = width * height * depth;

  LatticeCollision *collision = (LatticeCollision *)malloc(
      space.info.total_size * sizeof(LatticeCollision));
  LatticeNode *space_cpu =
      (LatticeNode *)malloc(space.info.total_size * sizeof(LatticeNode));

  for (int z = 0; z < depth; z++) {
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        size_t index = (z * width * height) + (y * width) + x;
        if (x == 0 || y == 0 || z == 0 || x == width - 1 || y == height - 1 ||
            z == depth - 1)
          collision[index] = LatticeCollisionEnum::BOUNCE_BACK_STATIC;
        else
          collision[index] = LatticeCollisionEnum::NO_COLLISION;

        for (int i = 0; i < LBM_SPEED_COUNTS; i++)
          space_cpu[index].f[i] = 0.f;
      }
    }
  }

  {
    int x = 3, y = 3, z = 3;
    size_t index = (z * width * height) + (y * width) + x;
    space_cpu[index].f[one_index] = 1.f;
  }

  lbm_space_init_device(&space, collision);
  lbm_space_copy_from_host(&space, space_cpu);
  free(collision);
  free(space_cpu);
  cuda_wait_for_device();
  return space;
}

float average_lbm_density(LatticeSpace &space) {
  float rho_avg = 0.f;
  for (size_t i = 0; i < space.info.total_size; i++)
    rho_avg += space.host_output[i].rho;
  rho_avg /= (float)space.info.total_size;
  return rho_avg;
}

std::string create_name(u_int32_t i) {
  return "./output/sample_" + std::to_string(i) + ".zip";
}

int main() {

  size_t width = 7, height = 7, depth = 7;
  int x = 3, y = 3, z = 3;
  size_t middle_index = (z * width * height) + (y * width) + x;

  for (int i = 0; i < LBM_SPEED_COUNTS; i++) {
    LatticeSpace space = create_test_experiment(i);
    for (int j = 0; j < 6; j++) {
      lbm_space_boundary_condition(&space);
      lbm_space_stream(&space);
    }
    LatticeNode *out = lbm_space_copy_to_host(&space);
    float val;
    if (i == 0)
      val = out[middle_index].f[i];
    else if (i & 1)
      val = out[middle_index].f[i + 1];
    else
      val = out[middle_index].f[i - 1];
    std::cout << "Index " << i;
    if (val == 1.f)
      std::cout << " passed." << std::endl;
    else
      std::cout << " failed." << std::endl;
    free(out);
    lbm_space_remove(&space);
  }

  return 0;

  /*
  DataCompressor compressor(11, 50);
  LatticeSpace space = create_cylinder_experiment();

  int sampling = 8000;
  for (u_int32_t i = 0; i < sampling; i++) {
    lbm_space_bgk_collision(&space);
    lbm_space_boundary_condition(&space);
    lbm_space_stream(&space);

    if (i % 10 == 0) {
      lbm_space_get_output(&space);
      compressor.save_memcpy_data((void *)space.host_output,
                                  space.info.total_size * sizeof(LatticeOutput),
                                  create_name(i));
      std::cout << i << "\t" << average_lbm_density(space) << "\t"
                << compressor.busy_threads() << std::endl;
    }
  }
  compressor.join();

  return 0;
  */
}
