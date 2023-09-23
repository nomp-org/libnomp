#include "nomp-test.h"

static int test_valid_arguments(const char *backend) {
  const char *argv[7] = {"foo.c", "--nomp-backend",  backend, "--nomp-device",
                         "0",     "--nomp-platform", "0"};
  int argc = 7;

  nomp_test_check(nomp_init(argc, argv));
  nomp_test_check(nomp_finalize());

  return 0;
}

static int test_ignore_non_argument_string(const char *backend) {
  const char *argv[8] = {"--nomp-backend",  backend, "cuda",
                         "--nomp-device",   "0",     "1",
                         "--nomp-platform", "0"};
  int argc = 8;

  nomp_test_check(nomp_init(argc, argv));
  nomp_test_check(nomp_finalize());

  return 0;
}

static int test_invalid_argument_flag(const char *backend) {
  const char *argv[8] = {"--nomp-backend",  backend, "--nomp-device", "0",
                         "--nomp-platform", "0",     "--unknown-arg", "value"};
  int argc = 8;

  nomp_test_check(nomp_init(argc, argv));
  nomp_test_check(nomp_finalize());

  return 0;
}

static int test_missing_argument(const char *backend) {
  const char *argv[5] = {"--nomp-backend", backend, "--nomp-device", "0",
                         "--nomp-platform"};
  int argc = 5;
  int err = nomp_init(argc, argv);
  nomp_test_assert(nomp_get_err_no(err) == NOMP_USER_INPUT_IS_INVALID);

  err = nomp_finalize();
  nomp_test_assert(err == NOMP_FINALIZE_FAILURE);
  nomp_test_assert(nomp_get_err_no(err) == NOMP_USER_LOG_ID_IS_INVALID);

  return 0;
}

int main(int argc, const char *argv[]) {
  char backend[NOMP_TEST_MAX_BUFFER_SIZE + 1];
  for (unsigned i = 0; i < (unsigned)argc; i++) {
    if (strncmp(argv[i], "--nomp-backend", NOMP_TEST_MAX_BUFFER_SIZE) == 0) {
      assert(i + 1 < (unsigned)argc);
      strncpy(backend, argv[i + 1], NOMP_TEST_MAX_BUFFER_SIZE);
      break;
    }
  }

  int err = 0;
  err |= SUBTEST(test_valid_arguments, backend);
  err |= SUBTEST(test_ignore_non_argument_string, backend);
  err |= SUBTEST(test_invalid_argument_flag, backend);
  err |= SUBTEST(test_missing_argument, backend);

  return err;
}
