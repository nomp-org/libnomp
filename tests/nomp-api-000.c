#include "nomp-test.h"

// Calling nomp_finalize() before nomp_init() should return an error.
static int test_first_nomp_finalize(void) {
  int err = nomp_finalize();
  nomp_test_assert(err == NOMP_FINALIZE_FAILURE);
  nomp_test_assert(nomp_get_log_no(err) == NOMP_USER_LOG_ID_IS_INVALID);
  nomp_test_assert(nomp_get_log_type(err) == NOMP_INVALID);

  return 0;
}

// Calling nomp_init() twice must return an error, but must not segfault.
static int test_nomp_init_twice(int argc, const char **argv) {
  nomp_test_check(nomp_init(argc, argv));

  int err = nomp_init(argc, argv);
  nomp_test_assert(nomp_get_log_no(err) == NOMP_INITIALIZE_FAILURE);

  char *desc = nomp_get_log_str(err);
  int eq = logcmp(desc, "\\[Error\\] .*libnomp\\/src\\/nomp.c:[0-9]* libnomp "
                        "is already initialized.");
  nomp_free(&desc);
  nomp_test_assert(eq);

  nomp_test_check(nomp_finalize());

  return 0;
}

// Calling nomp_finalize() twice must return an error, but must not segfault.
static int test_nomp_finalize_twice(int argc, const char **argv) {
  nomp_test_check(nomp_init(argc, argv));
  nomp_test_check(nomp_finalize());

  int err = nomp_finalize();
  nomp_test_assert(err == NOMP_FINALIZE_FAILURE);
  nomp_test_assert(nomp_get_log_no(err) == NOMP_USER_LOG_ID_IS_INVALID);
  nomp_test_assert(nomp_get_log_type(err) == NOMP_INVALID);

  return 0;
}

int main(int argc, const char *argv[]) {
  int err = 0;
  err |= SUBTEST(test_first_nomp_finalize);
  err |= SUBTEST(test_nomp_init_twice, argc, argv);
  err |= SUBTEST(test_nomp_finalize_twice, argc, argv);

  return err;
}
