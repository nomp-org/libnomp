#include "nomp-test.h"
#include "nomp.h"

// Strings not starting with "-" should be ignored.
static int test_ignore_non_option_string() {
  const char *argv[] = {"-b", "opencl", "cuda", "-d", "0", "1", "-p", "0"};
  int argc = 8;
  int err = nomp_init(argc, argv);
  nomp_test_chk(err);

  err = nomp_finalize();
  nomp_test_chk(err);

  return 0;
}

// FIXME: Non recognized options should be ignored. Currently, an error is
// returned which is wrong.
static int test_unrecognized_option() {
  const char *argv[] = {"--b", "opencl", "-d", "0", "-p", "0"};
  int argc = 6;
  int err = nomp_init(argc, argv);
  nomp_test_assert(nomp_get_log_no(err) == NOMP_USER_ARG_IS_INVALID);

  err = nomp_finalize();
  nomp_test_assert(nomp_get_log_no(err) == NOMP_RUNTIME_NOT_INITIALIZED);

  return 0;
}

// Recognized options missing the option value should return an error.
static int test_missing_option_value() {
  const char *argv[] = {"-b", "opencl", "-d", "0", "-p"};
  int argc = 5;
  int err = nomp_init(argc, argv);
  nomp_assert(nomp_get_log_no(err) == NOMP_USER_ARG_IS_INVALID);

  err = nomp_finalize();
  nomp_assert(nomp_get_log_no(err) == NOMP_RUNTIME_NOT_INITIALIZED);

  return 0;
}

// Valid options should succeed.
static int test_valid_options() {
  const char *argv[] = {"foo", "-b", "opencl", "-d", "0", "-p", "0"};
  int argc = 7;
  int err = nomp_init(argc, argv);
  nomp_test_chk(err);

  err = nomp_finalize();
  nomp_test_chk(err);

  return 0;
}

int main(int argc, const char *argv[]) {
  int err = SUBTEST(test_valid_options);
  err |= SUBTEST(test_ignore_non_option_string);
  err |= SUBTEST(test_unrecognized_option);
  err |= SUBTEST(test_missing_option_value);

  return err;
}
