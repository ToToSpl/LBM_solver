#ifndef LBM_CONSTANTS_INCLUDE
#define LBM_CONSTANTS_INCLUDE
#include "./lbm_gpu.cuh"

// HARDCODED CONSTANTS FOR NOW, TODO CHANGE TO STRUCT
#define CS2 1.f / 3.f
#define SIMULATION_DT 1.0f
#define SIMULATION_TAU 0.7f
#define SIMULATION_DT_TAU SIMULATION_DT / SIMULATION_TAU

// D3Q27 space see p.83 for more info
#define LBM_SPEED_COUNTS 27

typedef struct {
  float x, y, z;
} Vec3;

#define LBM_SPEED_VECTORS                                                      \
  {                                                                            \
    {0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {-1.f, 0.f, 0.f}, {0.f, 1.f, 0.f},       \
        {0.f, -1.f, 0.f}, {0.f, 0.f, 1.f}, {0.f, 0.f, -1.f}, {1.f, 1.f, 0.f},  \
        {-1.f, -1.f, 0.f}, {1.f, 0.f, 1.f}, {-1.f, 0.f, -1.f},                 \
        {0.f, 1.f, 1.f}, {0.f, -1.f, -1.f}, {1.f, -1.f, 0.f},                  \
        {-1.f, 1.f, 0.f}, {1.f, 0.f, -1.f}, {-1.f, 0.f, 1.f},                  \
        {0.f, 1.f, -1.f}, {0.f, -1.f, 1.f}, {1.f, 1.f, 1.f},                   \
        {-1.f, -1.f, -1.f}, {1.f, 1.f, -1.f}, {-1.f, -1.f, 1.f},               \
        {1.f, -1.f, 1.f}, {-1.f, 1.f, -1.f}, {-1.f, 1.f, 1.f}, {               \
      1.f, -1.f, -1.f                                                          \
    }                                                                          \
  }

#define LBM_SPEED_WEIGHTS                                                      \
  {                                                                            \
    8.f / 27.f, 2.f / 27.f, 2.f / 27.f, 2.f / 27.f, 2.f / 27.f, 2.f / 27.f,    \
        2.f / 27.f, 1.f / 54.f, 1.f / 54.f, 1.f / 54.f, 1.f / 54.f,            \
        1.f / 54.f, 1.f / 54.f, 1.f / 54.f, 1.f / 54.f, 1.f / 54.f,            \
        1.f / 54.f, 1.f / 54.f, 1.f / 54.f, 1.f / 216.f, 1.f / 216.f,          \
        1.f / 216.f, 1.f / 216.f, 1.f / 216.f, 1.f / 216.f, 1.f / 216.f,       \
        1.f / 216.f,                                                           \
  }

#endif
