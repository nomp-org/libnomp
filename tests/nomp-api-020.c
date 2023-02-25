#include "nomp-test.h"

static int test_valid_arguments() {
  const char *argv[7] = {"foo.c", "--nomp-backend",  "opencl", "--nomp-device",
                         "0",     "--nomp-platform", "0"};
  int argc = 7;

  nomp_test_chk(nomp_init(argc, argv));
  nomp_test_chk(nomp_finalize());

  return 0;
}

static int test_ignore_non_argument_string() {
  const char *argv[8] = {"--nomp-backend",  "opencl", "cuda",
                         "--nomp-device",   "0",      "1",
                         "--nomp-platform", "0"};
  int argc = 8;

  nomp_test_chk(nomp_init(argc, argv));
  nomp_test_chk(nomp_finalize());

  return 0;
}

static int test_invalid_argument_flag() {
  const char *argv[8] = {"--nomp-backend",  "opencl", "--nomp-device", "0",
                         "--nomp-platform", "0",      "--unknown-arg", "value"};
  int argc = 8;

  nomp_test_chk(nomp_init(argc, argv));
  nomp_test_chk(nomp_finalize());

  return 0;
}

static int test_missing_argument() {
  const char *argv[5] = {"--nomp-backend", "opencl", "--nomp-device", "0",
                         "--nomp-platform"};
  int argc = 5;
  int err = nomp_init(argc, argv);
  nomp_test_assert(nomp_get_log_no(err) == NOMP_USER_ARG_IS_INVALID);

  err = nomp_finalize();
  nomp_test_assert(nomp_get_log_no(err) == NOMP_FINALIZE_FAILURE);

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
