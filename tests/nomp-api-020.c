#include "nomp-test.h"

static int test_valid_arguments() {
  const char *argv[] = {"foo.c", "-b", "opencl", "-d", "0", "-p", "0"};
  int argc = 7;
  int err = nomp_init(argc, argv);
  nomp_test_chk(err);

  err = nomp_finalize();
  nomp_test_chk(err);

  return 0;
}

static int test_ignore_non_argument_string() {
  const char *argv[] = {"-b", "opencl", "cuda", "-d", "0", "1", "-p", "0"};
  int argc = 8;
  int err = nomp_init(argc, argv);
  nomp_test_chk(err);

  err = nomp_finalize();
  nomp_test_chk(err);

  return 0;
}

static int test_invalid_argument_flag() {
  const char *argv[] = {"--b", "opencl", "-d", "0", "-p", "0"};
  int argc = 6;
  int err = nomp_init(argc, argv);
  nomp_test_assert(nomp_get_log_no(err) == NOMP_USER_ARG_IS_INVALID);

  err = nomp_finalize();
  nomp_test_assert(nomp_get_log_no(err) == NOMP_RUNTIME_FINALIZE_FAILURE);

  return 0;
}

static int test_missing_argument() {
  const char *argv[] = {"-b", "opencl", "-d", "0", "-p"};
  int argc = 5;
  int err = nomp_init(argc, argv);
  nomp_test_assert(nomp_get_log_no(err) == NOMP_USER_ARG_IS_INVALID);

  err = nomp_finalize();
  nomp_test_assert(nomp_get_log_no(err) == NOMP_RUNTIME_FINALIZE_FAILURE);

  return 0;
}

int main(int argc, const char *argv[]) {
  int err = 0;
  err |= SUBTEST(test_valid_arguments);
  err |= SUBTEST(test_ignore_non_argument_string);
  err |= SUBTEST(test_invalid_argument_flag);
  err |= SUBTEST(test_missing_argument);

  return err;
}
