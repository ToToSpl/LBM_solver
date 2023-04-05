#include "./lbm_helpers.cuh"
#include "assert.h"

// Data taken from:
// https://stackoverflow.com/questions/5062781/cuda-max-threads-in-a-block
#define MAX_BLOCK_SIZE 1024
#define MAX_BLOCK_X 1024
#define MAX_BLOCK_Y 1024
#define MAX_BLOCK_Z 64
// Values choosen for best 2d and 3d alignment
#define DEFAULT_BLOCK_X 32
#define DEFAULT_BLOCK_Y 32
#define DEFAULT_BLOCK_Z 1

inline size_t ceil_grid(size_t a, size_t b) {
  // returns ceil of a / b
  size_t mod = a / b;
  size_t rest = a % b;
  if (rest == 0)
    return mod;
  else
    return mod + 1;
}

ComputeDim compute_dim_create(size_t x, size_t y, size_t z) {
  assert(x > 0 && y > 0 && z > 0);

  size_t total_size = x * y * z;

  // case where we can fit in one block
  if (total_size <= MAX_BLOCK_SIZE &&
      (x <= MAX_BLOCK_X && y <= MAX_BLOCK_Y && z <= MAX_BLOCK_Z)) {
    return {dim3(1, 1, 1), dim3(x, y, z)};
  }
  // otherwise divide space using default system
  size_t grid_x = ceil_grid(x, DEFAULT_BLOCK_X);
  size_t grid_y = ceil_grid(y, DEFAULT_BLOCK_Y);
  size_t grid_z = ceil_grid(z, DEFAULT_BLOCK_Z);

  size_t block_x = grid_x == 1 ? x : DEFAULT_BLOCK_X;
  size_t block_y = grid_y == 1 ? y : DEFAULT_BLOCK_Y;
  size_t block_z = grid_z == 1 ? z : DEFAULT_BLOCK_Z;

  return {dim3(grid_x, grid_y, grid_z), dim3(block_x, block_y, block_z)};
}
