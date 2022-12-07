#include "nomp-test.h"

// Calling `nomp_finalize` before `nomp_init` should return an error
static int test_first_nomp_finalize() {
  int output = 0;

  int err = nomp_finalize();
  nomp_test_assert(&output,
                   nomp_get_log_no(err) == NOMP_RUNTIME_NOT_INITIALIZED);

  char *desc;
  nomp_get_log_str(&desc, err);
  int matched = match_log(desc, "\\[Error\\] .*libnomp\\/src\\/nomp.c:[0-9]* "
                                "libnomp is not initialized.");
  nomp_test_assert(&output, matched);
  tfree(desc);
  return output;
}

// Calling `nomp_init` twice must return an error, but must not segfault
static int test_nomp_init_twice(char *backend, int device, int platform) {
  int output = 0;

  int err = nomp_init(backend, platform, device);
  nomp_test_chk(&output, err);
  err = nomp_init(backend, platform, device);
  nomp_test_assert(&output,
                   nomp_get_log_no(err) == NOMP_RUNTIME_ALREADY_INITIALIZED);

  char *desc;
  nomp_get_log_str(&desc, err);
  int matched =
      match_log(desc, "\\[Error\\] .*libnomp\\/src\\/nomp.c:[0-9]* libnomp is "
                      "already initialized to use opencl. Call nomp_finalize() "
                      "before calling nomp_init() again.");
  nomp_test_assert(&output, matched);
  tfree(desc);
  return output;
}

// Calling `nomp_finalize` twice must return an error, but must not segfault
static int test_nomp_finalize_twice() {
  int output = 0;

  int err = nomp_finalize();
  nomp_test_chk(&output, err);
  err = nomp_finalize();
  nomp_test_assert(&output,
                   nomp_get_log_no(err) == NOMP_RUNTIME_NOT_INITIALIZED);

  char *desc;
  nomp_get_log_str(&desc, err);
  int matched = match_log(desc, "\\[Error\\] .*libnomp\\/src\\/nomp.c:[0-9]* "
                                "libnomp is not initialized.");
  nomp_test_assert(&output, matched);
  tfree(desc);
  return output;
}

int main(int argc, char *argv[]) {
  char *backend;
  int device, platform;
  parse_input(argc, argv, &backend, &device, &platform);
  int err = 0;

  err |= SUBTEST(test_first_nomp_finalize);
  err |= SUBTEST(test_nomp_init_twice, backend, device, platform);
  err |= SUBTEST(test_nomp_finalize_twice);

  return err;
}
