#include <assert.h>
#include <math.h>
#include <nomp.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
  char *backend = argc > 1 ? argv[1] : "opencl";
  int device_id = argc > 2 ? atoi(argv[2]) : 0;
  int platform_id = argc > 3 ? atoi(argv[3]) : 0;

  // Calling `nomp_init` twice must return an error, but must not segfault
  int err = nomp_init(backend, device_id, platform_id);
  nomp_check_err(err);
  err = nomp_init(backend, device_id, platform_id);
  assert(err != 0);

  err = nomp_finalize();
  nomp_check_err(err);

  return err;
}
