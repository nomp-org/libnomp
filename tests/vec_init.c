#include <gnomp.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
  double a[10];

  int handle;
  gnomp_init(&handle, "opencl", 0, 0);

  gnomp_map(a, 0, 10, sizeof(double), GNOMP_H2D, handle);

  return 0;
}
