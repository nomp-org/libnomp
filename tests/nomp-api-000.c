#include "nomp.h"
#include <stdlib.h>

int main(int argc, char *argv[]) {
  char *backend = argc > 1 ? argv[1] : "opencl";
  int device = argc > 2 ? atoi(argv[2]) : 0;
  int platform = argc > 3 ? atoi(argv[3]) : 0;

  // Calling `nomp_finalize` before `nomp_init` should retrun an error
  int err = nomp_finalize();
  nomp_assert(err == NOMP_NOT_INITIALIZED_ERROR);

  // Calling `nomp_init` twice must return an error, but must not segfault
  err = nomp_init(backend, platform, device);
  nomp_chk(err);
  err = nomp_init(backend, platform, device);
  nomp_assert(err == NOMP_INITIALIZED_ERROR);

  // Calling `nomp_finalize` twice must return an error, but must not segfault
  err = nomp_finalize();
  nomp_chk(err);
  err = nomp_finalize();
  nomp_assert(err == NOMP_NOT_INITIALIZED_ERROR);

  return 0;
}
