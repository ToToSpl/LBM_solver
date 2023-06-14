// Include C++ header files.
#include <chrono>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <ostream>
#include <string>
#include <sys/types.h>

// Include local CUDA header files.
#include "include/lbm_constants.h"
#include "include/lbm_gpu.cuh"
#include "include/lbm_types.h"
#include "src/data_compressor.hpp"

#define STEPS 400000
#define SAVE_STEP 10000
#define DOT_WRITE 100

inline size_t get_index(size_t x, size_t y, size_t z, size_t w, size_t h) {
  return ((z * h + y) * w) + x;
}

LatticeSpace create_cylinder_experiment() {
  size_t width = 800, height = 400, depth = 100;
  size_t cyl_x = width / 4, cyl_y = height / 2,
         cyl_r2 = (height / 8) * (height / 8);
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
  {
    Vec3 u_in = {-0.06, 0.0f, 0.0f};
    Vec3 u_out = {0.06, 0.0f, 0.0f};

    space.info.wall_speeds.s1 = {u_in, InletDir::X_PLUS};
    space.info.wall_speeds.s2 = {u_out, InletDir::X_MINUS};
  }

  for (int z = 0; z < depth; z++) {
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        size_t index = get_index(x, y, z, width, height);
        for (int i = 0; i < LBM_SPEED_COUNTS; i++) {
          space_cpu[index].f[i] = 1.0f / LBM_SPEED_COUNTS;
        }
        space_cpu[index].f[1] += 0.06 / 2.0;
        space_cpu[index].f[2] -= 0.06 / 2.0;

        float r = z % 20 < 10 ? (55.0 * 55.0) : (45.0 * 45.0);

        if (y == 0 || y == height - 1)
          collision[index] = LatticeCollisionEnum::BOUNCE_BACK_STATIC;
        else if (z == 0 || z == depth - 1)
          collision[index] = LatticeCollisionEnum::WALL_ROLL;
        else if (x == 0)
          collision[index] = LatticeCollisionEnum::BOUNCE_BACK_SPEED_1;
        else if ((x - cyl_x) * (x - cyl_x) + (y - cyl_y) * (y - cyl_y) < r)
          collision[index] = LatticeCollisionEnum::BOUNCE_BACK_STATIC;
        else
          collision[index] = LatticeCollisionEnum::NO_COLLISION;
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

LatticeSpace create_box_experiment() {
  size_t width = 600, height = 200, depth = 200;
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
  {
    Vec3 u_in = {-0.05, 0.0f, 0.0f};
    Vec3 u_out = {0.05, 0.0f, 0.0f};

    space.info.wall_speeds.s1 = {u_in, InletDir::X_PLUS};
    space.info.wall_speeds.s2 = {u_out, InletDir::X_MINUS};
  }

  for (int z = 0; z < depth; z++) {
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        size_t index = get_index(x, y, z, width, height);
        for (int i = 0; i < LBM_SPEED_COUNTS; i++) {
          space_cpu[index].f[i] = 1.0f / LBM_SPEED_COUNTS;
        }
        space_cpu[index].f[1] += 0.05 / 2.0;
        space_cpu[index].f[2] -= 0.05 / 2.0;

        if (y == 0 || y == height - 1)
          collision[index] = LatticeCollisionEnum::BOUNCE_BACK_STATIC;
        else if (z == 0 || z == depth - 1)
          collision[index] = LatticeCollisionEnum::BOUNCE_BACK_STATIC;
        else if (x == 0)
          collision[index] = LatticeCollisionEnum::BOUNCE_BACK_SPEED_1;
        else if (x >= 100 && x < 200 && y <= 100 && z <= 100)
          collision[index] = LatticeCollisionEnum::BOUNCE_BACK_STATIC;
        else
          collision[index] = LatticeCollisionEnum::NO_COLLISION;
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

LatticeSpace create_airfoil_experiment() {
  size_t width = 6000, height = 4000, depth = 1;
  LatticeSpace space;
  space.info.x_size = width;
  space.info.y_size = height;
  space.info.z_size = depth;
  space.info.total_size = width * height * depth;

  std::ifstream file("airfoil.bin", std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Unable to open the file." << std::endl;
    exit(1);
  }
  file.seekg(0, std::ios::end);
  std::streampos fileSize = file.tellg();
  file.seekg(0, std::ios::beg);
  char *buffer = (char *)malloc(fileSize * sizeof(char));
  file.read(buffer, fileSize);

  LatticeCollision *collision = (LatticeCollision *)malloc(
      space.info.total_size * sizeof(LatticeCollision));
  LatticeNode *space_cpu =
      (LatticeNode *)malloc(space.info.total_size * sizeof(LatticeNode));

  // define inlet and outlet speed
  {
    Vec3 u_in = {-0.05, 0.0f, 0.0f};
    Vec3 u_out = {0.05, 0.0f, 0.0f};

    space.info.wall_speeds.s1 = {u_in, InletDir::X_PLUS};
    space.info.wall_speeds.s2 = {u_out, InletDir::X_MINUS};
  }

  for (int z = 0; z < depth; z++) {
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        size_t index = get_index(x, y, z, width, height);
        for (int i = 0; i < LBM_SPEED_COUNTS; i++) {
          space_cpu[index].f[i] = 1.0f / LBM_SPEED_COUNTS;
        }
        space_cpu[index].f[1] += 0.05 / 2.0;
        space_cpu[index].f[2] -= 0.05 / 2.0;

        if (y == 0 || y == height - 1)
          collision[index] = LatticeCollisionEnum::BOUNCE_BACK_MIRROR_XZ;
        else if (x == 0)
          collision[index] = LatticeCollisionEnum::BOUNCE_BACK_SPEED_1;
        else if (buffer[index])
          collision[index] = LatticeCollisionEnum::BOUNCE_BACK_STATIC;
        else
          collision[index] = LatticeCollisionEnum::NO_COLLISION;
      }
    }
  }

  lbm_space_init_device(&space, collision);
  free(collision);
  lbm_space_copy_from_host(&space, space_cpu);
  free(space_cpu);
  free(buffer);
  file.close();

  cuda_wait_for_device();
  return space;
}

std::string create_name(u_int32_t i) {
  return "./output/sample_" + std::to_string(i) + ".zip";
}

int main() {
  DataCompressor compressor(11, 50);
  // LatticeSpace space = create_cylinder_experiment();
  // LatticeSpace space = create_box_experiment();
  LatticeSpace space = create_airfoil_experiment();

  // int sampling = 5;
  for (u_int32_t i = 0; i <= STEPS; i++) {
    lbm_space_bgk_collision(&space);
    lbm_space_boundary_condition(&space);
    lbm_space_stream(&space);

    if (i % DOT_WRITE == 0)
      std::cout << "." << std::flush;
    if (i % SAVE_STEP == 0) {

#ifdef LBM_MOMENT_EXCHANGE
      lbm_space_calc_momentum(&space);
#endif
      lbm_space_get_output(&space);
      compressor.save_memcpy_data((void *)space.host_output,
                                  space.info.total_size * sizeof(LatticeOutput),
                                  create_name(i));
      std::cout << "\n"
                << i << "\t" << average_lbm_density(space) << "\t"
                << compressor.busy_threads() << std::endl;
    }
  }
  compressor.join();

  return 0;
}
