#ifndef LBM_DATA_HEADER
#define LBM_DATA_HEADER

#include <sys/types.h>

// D3Q27
// for directions and weights see p.90
typedef struct {
  float f[27];
} LatticeNode;

typedef struct {
  size_t x_size, y_size, z_size;
  size_t total_size;
} LatticeInfo;

typedef struct {
  float x, y, z;
} Vec3;

typedef struct {
  Vec3 u;
  float rho;
} LatticeOutput;

typedef struct {
  LatticeInfo info;
  LatticeNode *device_data;
  LatticeOutput *device_output;
} LatticeSpace;

void cuda_wait_for_device();
void lbm_space_init_device(LatticeSpace *space);
void lbm_space_init_kernel(LatticeSpace *space);
void lbm_space_bgk_collision(LatticeSpace *space);

void lbm_space_stream(LatticeSpace *space);

LatticeNode *lbm_space_copy_host(LatticeSpace *space);

#endif
