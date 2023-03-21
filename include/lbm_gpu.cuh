#ifndef LBM_DATA_HEADER
#define LBM_DATA_HEADER

#include "lbm_types.h"

void cuda_wait_for_device();
void lbm_space_init_device(LatticeSpace *space, LatticeCollision *collisions);
void lbm_space_init_kernel(LatticeSpace *space, float begin_spd_rho = 1.0f);
void lbm_space_bgk_collision(LatticeSpace *space);
void lbm_space_boundary_condition(LatticeSpace *space);
void lbm_space_get_output(LatticeSpace *space);

void lbm_space_stream(LatticeSpace *space);

LatticeNode *lbm_space_copy_host(LatticeSpace *space);

#endif
