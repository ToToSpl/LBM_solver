#ifndef LBM_DATA_HEADER
#define LBM_DATA_HEADER

#include <sys/types.h>

// D3Q27
// for directions and weights see p.90
typedef struct {
  float f[27];
} LatticeNode;

typedef struct {
  u_int32_t x_size, y_size, z_size;
  u_int32_t total_size;
} LatticeInfo;

typedef struct {
  LatticeInfo info;
  void *device_data; // cudaPitcherPtr
} LatticeSpace;

void lbm_space_init_device(LatticeSpace *space);
void lbm_space_init_kernel(LatticeSpace *space);

#endif
