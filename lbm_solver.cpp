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
  size_t width = 400, height = 100, depth = 1;
  size_t cyl_x = width / 4, cyl_y = height / 2,
         cyl_r2 = (height / 4) * (height / 4);
  LatticeSpace space;
  space.info.x_size = width;
  space.info.y_size = height;
  space.info.z_size = depth;
  space.info.total_size = width * height * depth;

  LatticeCollision *collision = (LatticeCollision *)malloc(
      space.info.total_size * sizeof(LatticeCollision));
  LatticeNode *space_cpu =
      (LatticeNode *)malloc(space.info.total_size * sizeof(LatticeNode));

  // define inlet and outlet speed
  Vec3 u_in = {-0.05f, 0.f, 0.f};
  space.info.wall_speeds.s1 = {u_in, InletDir::X_PLUS};  // inlet
  space.info.wall_speeds.s2 = {u_in, InletDir::X_MINUS}; // outlet

  for (int z = 0; z < depth; z++) {
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        size_t index = (z * width * height) + (y * width) + x;
        for (int i = 0; i < LBM_SPEED_COUNTS; i++) {
          space_cpu[index].f[i] = 1.0f;
        }
        space_cpu[index].f[1] += 0.5f;

        if (y == 0 || y == height - 1)
          collision[index] = LatticeCollisionEnum::BOUNCE_BACK_STATIC;
        // else if (x == 0)
        //   collision[index] = LatticeCollisionEnum::BOUNCE_BACK_SPEED_1;
        // else if (x == width - 1)
        //   collision[index] = LatticeCollisionEnum::BOUNCE_BACK_SPEED_2;
        else if ((x - cyl_x) * (x - cyl_x) + (y - cyl_y) * (y - cyl_y) < cyl_r2)
          collision[index] = LatticeCollisionEnum::BOUNCE_BACK_STATIC;
        else
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
  lbm_space_copy_from_host(&space, space_cpu);
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
}
