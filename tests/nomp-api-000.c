#include "nomp-test.h"
#include "nomp.h"

int main(int argc, char *argv[]) {
  char *backend;
  int device, platform;
  parse_input(argc, argv, &backend, &device, &platform);

  // Calling `nomp_finalize` before `nomp_init` should return an error
  int err = nomp_finalize();
  nomp_assert(nomp_get_log_no(err) == NOMP_RUNTIME_NOT_INITIALIZED);

  char *desc;
  err = nomp_get_log_str(&desc, err);
  int matched = match_log(desc, "\\[Error\\] .*libnomp\\/src\\/nomp.c:[0-9]* "
                                "libnomp is not initialized.");
  nomp_assert(matched);

  // Calling `nomp_init` twice must return an error, but must not segfault
  err = nomp_init(backend, platform, device);
  nomp_chk(err);
  err = nomp_init(backend, platform, device);
  nomp_assert(nomp_get_log_no(err) == NOMP_RUNTIME_ALREADY_INITIALIZED);

  err = nomp_get_log_str(&desc, err);
  matched =
      match_log(desc, "\\[Error\\] .*libnomp\\/src\\/nomp.c:[0-9]* libnomp is "
                      "already initialized to use opencl. Call nomp_finalize() "
                      "before calling nomp_init() again.");
  nomp_assert(matched);

  // Calling `nomp_finalize` twice must return an error, but must not segfault
  err = nomp_finalize();
  nomp_chk(err);
  err = nomp_finalize();
  nomp_assert(nomp_get_log_no(err) == NOMP_RUNTIME_NOT_INITIALIZED);

  err = nomp_get_log_str(&desc, err);
  matched = match_log(desc, "\\[Error\\] .*libnomp\\/src\\/nomp.c:[0-9]* "
                            "libnomp is not initialized.");
  nomp_assert(matched);

  tfree(desc);

  return 0;
}
