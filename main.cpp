// Include C++ header files.
#include <cstddef>
#include <iostream>
#include <sys/types.h>

#include <chrono>

// Include local CUDA header files.
#include "include/lbm_constants.h"
#include "include/lbm_gpu.cuh"

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
  LatticeWall inlet = {u_in, InletDir::X_PLUS};
  LatticeWall outlet = {u_in, InletDir::X_MINUS};
  space.info.wall_speeds.s1 = inlet;
  space.info.wall_speeds.s2 = outlet;

  for (int z = 0; z < depth; z++) {
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        size_t index = (z * width * height) + (y * width) + x;

        // inlet
        if (x == 0)
          collision[index] = LatticeCollisionEnum::BOUNCE_BACK_SPEED_1;
        // outlet
        else if (x == space.info.x_size - 1)
          collision[index] = LatticeCollisionEnum::BOUNCE_BACK_SPEED_2;
        // side walls
        else if (y == 0 || y == height - 1 || z == 0 || z == depth - 1)
          collision[index] = LatticeCollisionEnum::BOUNCE_BACK_STATIC;
        // cylinder
        else if ((x - cyl_x) * (x - cyl_x) + (y - cyl_y) * (y - cyl_y) < cyl_r2)
          collision[index] = LatticeCollisionEnum::BOUNCE_BACK_STATIC;
        // normal space
        else
          collision[index] = LatticeCollisionEnum::NO_COLLISION;
      }
    }
  }

  lbm_space_init_device(&space, collision);
  free(collision);

  lbm_space_init_kernel(&space, 1.0f);
  cuda_wait_for_device();
  return space;
}

int main() {
  LatticeSpace space = create_cylinder_experiment();

  auto start = std::chrono::high_resolution_clock::now();
  int sampling = 10;
  for (int i = 0; i < sampling; i++) {
    lbm_space_bgk_collision(&space);
    lbm_space_boundary_condition(&space);
    lbm_space_stream(&space);
    lbm_space_get_output(&space, nullptr);
  }
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  std::cout << (float)duration.count() / ((float)sampling * 1000.f) << "s"
            << std::endl;

  return 0;
}
