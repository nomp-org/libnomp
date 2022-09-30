#include "nomp.h"
#include <stdlib.h>

int main(int argc, char *argv[]) {
  char *backend = argc > 1 ? argv[1] : "opencl";
  int device_id = argc > 2 ? atoi(argv[2]) : 0;
  int platform_id = argc > 3 ? atoi(argv[3]) : 0;

  int err = nomp_init(backend, device_id, platform_id);
  nomp_chk(err);

  int a[10] = {0}, b[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

  // Free'ing before mapping should return an error
  err = nomp_update(a, 0, 10, sizeof(int), NOMP_FREE);
  nomp_assert(err == NOMP_INVALID_MAP_PTR);

  // D2H before H2D should return an error
  err = nomp_update(a, 0, 10, sizeof(int), NOMP_FROM);
  nomp_assert(err == NOMP_INVALID_MAP_PTR);

  // Mapping H2D multiple times is not an error
  err = nomp_update(b, 0, 10, sizeof(int), NOMP_TO);
  nomp_chk(err);
  err = nomp_update(b, 0, 10, sizeof(int), NOMP_TO);
  nomp_chk(err);

  err = nomp_update(a, 0, 10, sizeof(int), NOMP_TO);
  nomp_chk(err);

  // Mapping D2H multiple times is not an error
  err = nomp_update(a, 0, 10, sizeof(int), NOMP_FROM);
  nomp_chk(err);
  err = nomp_update(a, 0, 10, sizeof(int), NOMP_FROM);
  nomp_chk(err);

  // Set b to all 0's, then copy the value on the device and
  // check if it is all 1's
  for (int i = 0; i < 10; i++)
    b[i] = 0;
  err = nomp_update(b, 0, 10, sizeof(int), NOMP_FROM);
  nomp_chk(err);
  for (int i = 0; i < 10; i++)
    nomp_assert(b[i] == 1);

  // Free'ing after mapping is not an error
  err = nomp_update(a, 0, 10, sizeof(int), NOMP_FREE);
  nomp_chk(err);
  err = nomp_update(b, 0, 10, sizeof(int), NOMP_FREE);
  nomp_chk(err);

  err = nomp_finalize();
  nomp_chk(err);

  return 0;
}
