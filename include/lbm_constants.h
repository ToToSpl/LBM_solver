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

// indecies which needs to be streamed in given direction
#define LBM_STREAM_X_PLUS                                                      \
  { 1, 7, 9, 13, 15, 19, 21, 23, 26 }
#define LBM_STREAM_X_MINUS                                                     \
  { 2, 8, 10, 14, 16, 20, 22, 24, 25 }
#define LBM_STREAM_Y_PLUS                                                      \
  { 3, 7, 11, 14, 17, 19, 21, 24, 25 }
#define LBM_STREAM_Y_MINUS                                                     \
  { 4, 8, 12, 13, 18, 20, 22, 23, 26 }
#define LBM_STREAM_Z_PLUS                                                      \
  { 5, 9, 11, 16, 18, 19, 22, 23, 25 }
#define LBM_STREAM_Z_MINUS                                                     \
  { 6, 10, 12, 15, 17, 20, 21, 24, 26 }

// TODO: can also be described as: odd -> index+1, even -> index-1. check which
// is faster
#define LBM_COLLISION_MIRROR                                                   \
  {                                                                            \
    0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19,  \
        22, 21, 24, 23, 26, 25                                                 \
  }

#endif
