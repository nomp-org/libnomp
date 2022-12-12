#include "nomp-test.h"
#include "nomp.h"

static int test_valid_arguments(int argc, const char **argv) {
  int err = nomp_init(argc, argv);
  nomp_test_chk(err);
  err = nomp_finalize();
  nomp_test_chk(err);
  return 0;
}

static int test_invalid_arguments(int argc, const char **argv) {
  int err = nomp_init(argc, argv);
  nomp_test_chk(err);
  err = nomp_finalize();
  nomp_test_chk(err);
  return 0;
}

static int test_invalid_argument_flag(int argc, const char **argv) {
  int err = nomp_init(argc, argv);
  nomp_test_assert(nomp_get_log_no(err) == NOMP_USER_ARGS_IS_INVALID);
  err = nomp_finalize();
  nomp_test_assert(nomp_get_log_no(err) == NOMP_RUNTIME_NOT_INITIALIZED);
  return 0;
}

static int test_missing_argument(int argc, const char **argv) {
  int err = nomp_init(argc, argv);
  nomp_assert(nomp_get_log_no(err) == NOMP_USER_ARGS_IS_INVALID);
  err = nomp_finalize();
  nomp_assert(nomp_get_log_no(err) == NOMP_RUNTIME_NOT_INITIALIZED);
  return 0;
}

int main(int argc, const char *argv[]) {

  int err = 0;

  const char *args1[] = {"foo.c", "-b", "opencl", "-d", "0", "-p", "0"};
  argc = 7;
  err |= SUBTEST(test_valid_arguments, argc, args1);

  const char *args2[] = {"-b", "opencl", "cuda", "-d", "0", "1", "-p", "0"};
  argc = 8;
  err |= SUBTEST(test_invalid_arguments, argc, args2);

  const char *args3[] = {"--b", "opencl", "-d", "0", "-p", "0"};
  argc = 6;
  err |= SUBTEST(test_invalid_argument_flag, argc, args3);

  const char *args4[] = {"-b", "opencl", "-d", "0", "-p"};
  argc = 5;
  err |= SUBTEST(test_missing_argument, argc, args4);

  return 0;
}
