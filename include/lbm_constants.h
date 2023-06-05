#ifndef LBM_CONSTANTS_INCLUDE
#define LBM_CONSTANTS_INCLUDE

// HARDCODED CONSTANTS FOR NOW, TODO CHANGE TO STRUCT
#define CS2 (1.f / 3.f)
#define SIMULATION_DT 1.0f
#define SIMULATION_TAU 0.6f
#define SIMULATION_DT_TAU (SIMULATION_DT / SIMULATION_TAU)

// #define D3Q27_VERSION
#define D2Q9_VERSION

#ifdef D3Q27_VERSION
// D3Q27 space see p.83 for more info
#define LBM_SPEED_COUNTS 27

// clang-format off
#define LBM_SPEED_VECTORS                                                       \
  {                                                                             \
    {0.f, 0.f, 0.f},                                                            \
    {1.f, 0.f, 0.f},                                                            \
    {-1.f, 0.f, 0.f},                                                           \
    {0.f, 1.f, 0.f},                                                            \
    {0.f, -1.f, 0.f},                                                           \
    {0.f, 0.f, 1.f},                                                            \
    {0.f, 0.f, -1.f},                                                           \
    {1.f, 1.f, 0.f},                                                            \
    {-1.f, -1.f, 0.f},                                                          \
    {1.f, 0.f, 1.f},                                                            \
    {-1.f, 0.f, -1.f},                                                          \
    {0.f, 1.f, 1.f},                                                            \
    {0.f, -1.f, -1.f},                                                          \
    {1.f, -1.f, 0.f},                                                           \
    {-1.f, 1.f, 0.f},                                                           \
    {1.f, 0.f, -1.f},                                                           \
    {-1.f, 0.f, 1.f},                                                           \
    {0.f, 1.f, -1.f},                                                           \
    {0.f, -1.f, 1.f},                                                           \
    {1.f, 1.f, 1.f},                                                            \
    {-1.f, -1.f, -1.f},                                                         \
    {1.f, 1.f, -1.f},                                                           \
    {-1.f, -1.f, 1.f},                                                          \
    {1.f, -1.f, 1.f},                                                           \
    {-1.f, 1.f, -1.f},                                                          \
    {-1.f, 1.f, 1.f},                                                           \
    {1.f, -1.f, -1.f}}

#define LBM_SPEED_WEIGHTS                                                       \
  {                                                                             \
    8.f / 27.f,                                                                 \
    2.f / 27.f,                                                                 \
    2.f / 27.f,                                                                 \
    2.f / 27.f,                                                                 \
    2.f / 27.f,                                                                 \
    2.f / 27.f,                                                                 \
    2.f / 27.f,                                                                 \
    1.f / 54.f,                                                                 \
    1.f / 54.f,                                                                 \
    1.f / 54.f,                                                                 \
    1.f / 54.f,                                                                 \
    1.f / 54.f,                                                                 \
    1.f / 54.f,                                                                 \
    1.f / 54.f,                                                                 \
    1.f / 54.f,                                                                 \
    1.f / 54.f,                                                                 \
    1.f / 54.f,                                                                 \
    1.f / 54.f,                                                                 \
    1.f / 54.f,                                                                 \
    1.f / 216.f,                                                                \
    1.f / 216.f,                                                                \
    1.f / 216.f,                                                                \
    1.f / 216.f,                                                                \
    1.f / 216.f,                                                                \
    1.f / 216.f,                                                                \
    1.f / 216.f,                                                                \
    1.f / 216.f,                                                                \
  }
// clang-format on

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

#define LBM_BOUNCE_BACK_MIRROR_XZ                                              \
  {                                                                            \
    0, 1, 2, 4, 3, 5, 6, 13, 14, 9, 10, 18, 17, 7, 8, 15, 16, 12, 11, 23, 24,  \
        26, 25, 19, 20, 22, 21                                                 \
  }

#define LBM_BOUNCE_BACK_MIRROR_YZ                                              \
  {                                                                            \
    0, 2, 1, 3, 4, 5, 6, 14, 13, 16, 15, 11, 12, 8, 7, 10, 9, 17, 18, 25, 26,  \
        24, 23, 22, 21, 19, 20                                                 \
  }

#define LBM_BOUNCE_BACK_MIRROR_XY                                              \
  {                                                                            \
    0, 1, 2, 3, 4, 6, 5, 7, 8, 15, 16, 17, 18, 13, 14, 9, 10, 11, 12, 21, 22,  \
        19, 20, 26, 25, 24, 23                                                 \
  }

#endif
#ifdef D2Q9_VERSION

#define LBM_SPEED_COUNTS 9

// clang-format off
#define LBM_SPEED_VECTORS                                                       \
  {                                                                             \
    {0.f, 0.f, 0.f},                                                            \
    {1.f, 0.f, 0.f},                                                            \
    {-1.f, 0.f, 0.f},                                                           \
    {0.f, 1.f, 0.f},                                                            \
    {0.f, -1.f, 0.f},                                                           \
    {1.f, 1.f, 0.f},                                                            \
    {-1.f, -1.f, 0.f},                                                          \
    {1.f, -1.f, 0.f},                                                           \
    {-1.f, 1.f, 0.f}                                                            \
  }

#define LBM_SPEED_WEIGHTS                                                       \
  {                                                                             \
    4.f / 9.f,                                                                  \
    1.f / 9.f,                                                                  \
    1.f / 9.f,                                                                  \
    1.f / 9.f,                                                                  \
    1.f / 9.f,                                                                  \
    1.f / 36.f,                                                                 \
    1.f / 36.f,                                                                 \
    1.f / 36.f,                                                                 \
    1.f / 36.f,                                                                 \
  }
// clang-format on

// indecies which needs to be streamed in given direction
#define LBM_STREAM_X_PLUS                                                      \
  { 1, 5, 7 }
#define LBM_STREAM_X_MINUS                                                     \
  { 2, 6, 8 }
#define LBM_STREAM_Y_PLUS                                                      \
  { 3, 5, 8 }
#define LBM_STREAM_Y_MINUS                                                     \
  { 4, 6, 7 }
#define LBM_STREAM_Z_PLUS                                                      \
  { 0 }
#define LBM_STREAM_Z_MINUS                                                     \
  { 0 }

// Mirror along xz plane
#define LBM_BOUNCE_BACK_MIRROR_XZ                                              \
  { 0, 1, 2, 4, 3, 7, 8, 5, 6 }

// Mirror along yz plane
#define LBM_BOUNCE_BACK_MIRROR_YZ                                              \
  { 0, 2, 1, 3, 4, 8, 7, 6, 5 }

// Mirror along z plane, do nothing
#define LBM_BOUNCE_BACK_MIRROR_XY                                              \
  { 0, 1, 2, 3, 4, 5, 6, 7, 8 }

#endif

#endif
