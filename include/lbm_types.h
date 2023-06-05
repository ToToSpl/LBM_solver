#ifndef LBM_TYPES_INCLUDE
#define LBM_TYPES_INCLUDE

#include <sys/types.h>

enum LatticeCollisionEnum {
  NO_COLLISION = 0,       // do nothing
  WALL_ROLL = 0,          // rolls to other side
  BOUNCE_BACK_STATIC = 1, // collision with no moving wall
  // for now supports only two different speeds
  BOUNCE_BACK_SPEED_1 = 2, // collision with moving wall, speed set in struct
  BOUNCE_BACK_SPEED_2 = 3, // collision with moving wall, speed set in struct
  // collision with mirror plane
  BOUNCE_BACK_MIRROR_X = 4,
  BOUNCE_BACK_MIRROR_Y = 5,
  BOUNCE_BACK_MIRROR_Z = 6,
};

typedef u_int8_t LatticeCollision;
// D3Q27
// for directions and weights see p.90
typedef struct {
  float f[27];
} LatticeNode;

typedef struct {
  float x, y, z;
} Vec3;

enum InletDir { X_PLUS, X_MINUS, Y_PLUS, Y_MINUS, Z_PLUS, Z_MINUS };
typedef struct {
  Vec3 u;
  InletDir dir;
} LatticeWall;

typedef struct {
  LatticeWall s1, s2;
} LatticeWallSpeeds;

typedef struct {
  Vec3 u;
  float rho;
} LatticeOutput;

typedef struct {
  size_t x_size, y_size, z_size;
  size_t total_size;
  LatticeWallSpeeds wall_speeds;
} LatticeInfo;

typedef struct {
  LatticeInfo info;
  LatticeNode *device_data;
  LatticeCollision *device_collision;
  LatticeOutput *device_output;
  LatticeOutput *host_output;
} LatticeSpace;

#endif
