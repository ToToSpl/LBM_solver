#ifndef LBM_HELPERS_INCLUDE
#define LBM_HELPERS_INCLUDE
#include <cuda.h>

typedef struct {
  dim3 gridSize, blockSize;
} ComputeDim;

ComputeDim compute_dim_create(size_t x, size_t y, size_t z);

#endif
