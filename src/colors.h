#ifndef COLORS_INCLUDE
#define COLORS_INCLUDE
#include <math.h>

typedef struct {
  double r; // a fraction between 0 and 1
  double g; // a fraction between 0 and 1
  double b; // a fraction between 0 and 1
} rgb;

typedef struct {
  double h; // angle in radians
  double s; // a fraction between 0 and 1
  double v; // a fraction between 0 and 1
} hsv;

hsv rgb2hsv(rgb in);
rgb hsv2rgb(hsv in);

#endif
