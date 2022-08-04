#include "nomp.h"
#include <stdlib.h>

int main(int argc, char *argv[]) {
  char *backend = argc > 1 ? argv[1] : "opencl";
  int device_id = argc > 2 ? atoi(argv[2]) : 0;
  int platform_id = argc > 3 ? atoi(argv[3]) : 0;

  int err = nomp_init(backend, device_id, platform_id);
  nomp_chk(err);

  int a[10] = {0}, b[10] = {1};

  // Free'ing before mapping should return an error
  err = nomp_map(a, 0, 10, sizeof(int), NOMP_FREE);
  nomp_assert(err == NOMP_INVALID_MAP_PTR);

  // D2H before H2D should return an error
  err = nomp_map(a, 0, 10, sizeof(int), NOMP_D2H);
  nomp_assert(err == NOMP_INVALID_MAP_PTR);

  // Mapping H2D multiple times is not an error
  err = nomp_map(b, 0, 10, sizeof(int), NOMP_H2D);
  nomp_chk(err);
  err = nomp_map(b, 0, 10, sizeof(int), NOMP_H2D);
  nomp_chk(err);

  err = nomp_map(a, 0, 10, sizeof(int), NOMP_H2D);
  nomp_chk(err);

  // Mapping D2H multiple times is not an error
  err = nomp_map(a, 0, 10, sizeof(int), NOMP_D2H);
  nomp_chk(err);
  err = nomp_map(a, 0, 10, sizeof(int), NOMP_D2H);
  nomp_chk(err);

  // Free'ing after mapping is not an error
  err = nomp_map(a, 0, 10, sizeof(int), NOMP_FREE);
  nomp_chk(err);
  err = nomp_map(b, 0, 10, sizeof(int), NOMP_FREE);
  nomp_chk(err);

  err = nomp_finalize();
  nomp_chk(err);

  return 0;
}
