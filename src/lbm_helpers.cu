#include "./lbm_helpers.cuh"
#include "assert.h"

#define INTEGER_DIV_CEIL(X, Y) ((X + (Y + 1)) / Y)

ComputeDim compute_dim_create(size_t x, size_t y, size_t z) {
  assert(x > 0 && y > 0 && z > 0);

  size_t total_size = x * y * z;

  // case where we can fit in one block
  if (total_size <= MAX_BLOCK_SIZE &&
      (x <= MAX_BLOCK_X && y <= MAX_BLOCK_Y && z <= MAX_BLOCK_Z)) {
    return {dim3(1, 1, 1), dim3(x, y, z)};
  }
  // otherwise divide space using default system
  size_t grid_x = INTEGER_DIV_CEIL(x, DEFAULT_BLOCK_X);
  size_t grid_y = INTEGER_DIV_CEIL(x, DEFAULT_BLOCK_Y);
  size_t grid_z = INTEGER_DIV_CEIL(x, DEFAULT_BLOCK_Z);

  return {dim3(grid_x, grid_y, grid_z),
          dim3(DEFAULT_BLOCK_X, DEFAULT_BLOCK_Y, DEFAULT_BLOCK_Z)};
}
