#include "nomp-test.h"
#include <limits.h>

// NOMP_nomp-backend environment variable with invalid value.
static int test_invalid_nomp_backend(int argc, const char **argv) {
  setenv("NOMP_BACKEND", "invalid", 1);
  nomp_test_assert(nomp_get_log_no(nomp_init(argc, argv)) ==
                   NOMP_USER_INPUT_IS_INVALID);
  unsetenv("NOMP_BACKEND");

  nomp_test_assert(nomp_get_log_no(nomp_finalize()) == NOMP_FINALIZE_FAILURE);

  return 0;
}

// NOMP_PLATFORM environment variable with invalid value.
static int test_invalid_platform_id(int argc, const char **argv) {
  setenv("NOMP_PLATFORM", "invalid", 1);
  nomp_test_assert(nomp_get_log_no(nomp_init(argc, argv)) ==
                   NOMP_USER_INPUT_IS_INVALID);
  unsetenv("NOMP_PLATFORM");

  nomp_test_assert(nomp_get_log_no(nomp_finalize()) == NOMP_FINALIZE_FAILURE);

  return 0;
}

// NOMP_DEVICE environment variable with invalid value.
static int test_invalid_device_id(int argc, const char **argv) {
  setenv("NOMP_DEVICE", "invalid", 1);
  nomp_test_assert(nomp_get_log_no(nomp_init(argc, argv)) ==
                   NOMP_USER_INPUT_IS_INVALID);
  unsetenv("NOMP_DEVICE");

  nomp_test_assert(nomp_get_log_no(nomp_finalize()) == NOMP_FINALIZE_FAILURE);

  return 0;
}

// NOMP_VERBOSE environment variable with invalid value.
static int test_invalid_nomp_verbose(int argc, const char **argv) {
  setenv("NOMP_VERBOSE", "4", 1);
  int err = nomp_init(argc, argv);
  unsetenv("NOMP_VERBOSE");

  nomp_test_assert(nomp_get_log_no(err) == NOMP_USER_INPUT_IS_INVALID);
  char *desc = nomp_get_log_str(err);
  int eq = logcmp(
      desc, "\\[Error\\] .*libnomp\\/src\\/log.c:[0-9]* Invalid verbose level "
            "4 is provided. The value should be within the range 0-3.");
  nomp_test_assert(eq);
  nomp_free(&desc);

  nomp_test_assert(nomp_get_log_no(nomp_finalize()) == NOMP_FINALIZE_FAILURE);

  return 0;
}

// Run with a valid NOMP_nomp-backend environment variable.
static int test_valid_nomp_backend(int argc, const char **argv) {
  setenv("NOMP_BACKEND", "opencl", 1);
  nomp_test_chk(nomp_init(argc, argv));
  unsetenv("NOMP_BACKEND");

  nomp_test_chk(nomp_finalize());

  return 0;
}

// NOMP_nomp-backend value is not case sensitive.
static int test_nomp_backend_case_insensitivity(int argc, const char **argv) {
  setenv("NOMP_BACKEND", "oPenCl", 1);
  nomp_test_chk(nomp_init(argc, argv));
  unsetenv("NOMP_BACKEND");

  nomp_test_chk(nomp_finalize());

  return 0;
}

// Run with a valid NOMP_PLATFORM environment variable.
static int test_valid_platform_id(int argc, const char **argv) {
  setenv("NOMP_PLATFORM", "0", 1);
  nomp_test_chk(nomp_init(argc, argv));
  unsetenv("NOMP_PLATFORM");

  nomp_test_chk(nomp_finalize());

  return 0;
}

// Run with a valid NOMP_DEVICE  environment variable.
static int test_valid_device_id(int argc, const char **argv) {
  setenv("NOMP_DEVICE", "0", 1);
  nomp_test_chk(nomp_init(argc, argv));
  unsetenv("NOMP_DEVICE");

  nomp_test_chk(nomp_finalize());

  return 0;
}

int main(int argc, const char *argv[]) {
  int err = 0, argsc = 6;
  const char *args0[6] = {"--nomp-backend",  "opencl", "--nomp-device", "0",
                          "--nomp-platform", "0"};
  err |= SUBTEST(test_invalid_nomp_backend, argsc, args0);
  err |= SUBTEST(test_invalid_platform_id, argsc, args0);
  err |= SUBTEST(test_invalid_device_id, argsc, args0);
  err |= SUBTEST(test_invalid_nomp_verbose, argsc, args0);

  const char *args1[6] = {"--nomp-backend",  "invalid", "--nomp-device", "0",
                          "--nomp-platform", "0"};
  err |= SUBTEST(test_valid_nomp_backend, argsc, args1);
  err |= SUBTEST(test_nomp_backend_case_insensitivity, argsc, args1);

  char *max_int = nomp_calloc(char, 100);
  snprintf(max_int, 100, "%d", INT_MAX);

  const char *args2[6] = {"--nomp-backend",  "opencl", "--nomp-device", "0",
                          "--nomp-platform", max_int};
  err |= SUBTEST(test_valid_platform_id, argsc, args2);

  const char *args3[6] = {"--nomp-backend", "opencl",          "--nomp-device",
                          max_int,          "--nomp-platform", "0"};
  err |= SUBTEST(test_valid_device_id, argsc, args3);

  nomp_free(&max_int);

  return err;
}
