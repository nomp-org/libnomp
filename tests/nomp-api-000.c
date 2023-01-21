#include "nomp-test.h"

// Calling nomp_finalize() before nomp_init() should return an error.
static int test_first_nomp_finalize() {
  int err = nomp_finalize();
  nomp_test_assert(nomp_get_log_no(err) == NOMP_RUNTIME_NOT_INITIALIZED);

  char *desc;
  nomp_get_log_str(&desc, err);
  int matched = match_log(desc, "\\[Error\\] .*libnomp\\/src\\/nomp.c:[0-9]* "
                                "libnomp is not initialized.");
  tfree(desc);
  nomp_test_assert(matched);

  return 0;
}

// Calling nomp_init() twice must return an error, but must not segfault.
static int test_nomp_init_twice(int argc, const char **argv) {
  int err = nomp_init(argc, argv);
  nomp_test_chk(err);
  err = nomp_init(argc, argv);
  nomp_test_assert(nomp_get_log_no(err) == NOMP_RUNTIME_ALREADY_INITIALIZED);

  char *desc;
  nomp_get_log_str(&desc, err);
  int matched =
      match_log(desc, "\\[Error\\] .*libnomp\\/src\\/nomp.c:[0-9]* libnomp is "
                      "already initialized to use opencl. Call nomp_finalize() "
                      "before calling nomp_init() again.");
  tfree(desc);
  nomp_test_assert(matched);

  return 0;
}

// Calling nomp_finalize() twice must return an error, but must not segfault.
static int test_nomp_finalize_twice() {
  int err = nomp_finalize();
  nomp_test_chk(err);
  err = nomp_finalize();
  nomp_test_assert(nomp_get_log_no(err) == NOMP_RUNTIME_NOT_INITIALIZED);

  char *desc;
  nomp_get_log_str(&desc, err);
  int matched = match_log(desc, "\\[Error\\] .*libnomp\\/src\\/nomp.c:[0-9]* "
                                "libnomp is not initialized.");
  tfree(desc);
  nomp_test_assert(matched);

  return 0;
}

int main(int argc, const char *argv[]) {
  int err = 0;
  err |= SUBTEST(test_first_nomp_finalize);
  err |= SUBTEST(test_nomp_init_twice, argc, argv);
  err |= SUBTEST(test_nomp_finalize_twice);

  return err;
}
