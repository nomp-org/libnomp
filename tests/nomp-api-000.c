#include "nomp-test.h"
#include "nomp.h"

int main(int argc, char *argv[]) {
  char *backend;
  int device, platform;
  parse_input(argc, argv, &backend, &device, &platform);

  // Calling `nomp_finalize` before `nomp_init` should retrun an error
  int err = nomp_finalize();
  nomp_assert(nomp_get_log_no(err) == NOMP_RUNTIME_NOT_INITIALIZED);

  // Calling `nomp_init` twice must return an error, but must not segfault
  err = nomp_init(backend, platform, device);
  nomp_chk(err);
  err = nomp_init(backend, platform, device);
  nomp_assert(nomp_get_log_no(err) == NOMP_RUNTIME_ALREADY_INITIALIZED);

  // Calling `nomp_finalize` twice must return an error, but must not segfault
  err = nomp_finalize();
  nomp_chk(err);
  err = nomp_finalize();
  nomp_assert(nomp_get_log_no(err) == NOMP_RUNTIME_NOT_INITIALIZED);

  return 0;
}
