#ifndef CUDA_HELPERS_INCLUDE
#define CUDA_HELPERS_INCLUDE
#include <cuda.h>

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

typedef struct {
  dim3 gridSize, blockSize;
} ComputeDim;

ComputeDim compute_dim_create(size_t x, size_t y, size_t z);

#endif
