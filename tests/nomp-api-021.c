#include "nomp-test.h"
#include <limits.h>

// NOMP_BACKEND environment variable with invalid value.
static int test_invalid_backend(int argc, const char **argv) {
  setenv("NOMP_BACKEND", "invalid", 1);
  int err = nomp_init(argc, argv);
  unsetenv("NOMP_BACKEND");

  nomp_test_assert(nomp_get_log_no(err) == NOMP_USER_INPUT_IS_INVALID);

  err = nomp_finalize();
  nomp_test_assert(nomp_get_log_no(err) == NOMP_FINALIZE_FAILURE);

  return 0;
}

// NOMP_PLATFORM_ID environment variable with invalid value.
static int test_invalid_platform_id(int argc, const char **argv) {
  setenv("NOMP_PLATFORM_ID", "invalid", 1);
  int err = nomp_init(argc, argv);
  unsetenv("NOMP_PLATFORM_ID");

  nomp_test_assert(nomp_get_log_no(err) == NOMP_USER_INPUT_IS_INVALID);

  err = nomp_finalize();
  nomp_test_assert(nomp_get_log_no(err) == NOMP_FINALIZE_FAILURE);

  return 0;
}

// NOMP_DEVICE_ID environment variable with invalid value.
static int test_invalid_device_id(int argc, const char **argv) {
  setenv("NOMP_DEVICE_ID", "invalid", 1);
  int err = nomp_init(argc, argv);
  unsetenv("NOMP_DEVICE_ID");

  nomp_test_assert(nomp_get_log_no(err) == NOMP_USER_INPUT_IS_INVALID);

  err = nomp_finalize();
  nomp_test_assert(nomp_get_log_no(err) == NOMP_FINALIZE_FAILURE);

  return 0;
}

// Run with a valid NOMP_BACKEND environment variable.
static int test_valid_backend(int argc, const char **argv) {
  setenv("NOMP_BACKEND", "opencl", 1);
  int err = nomp_init(argc, argv);
  unsetenv("NOMP_BACKEND");

  nomp_test_chk(err);

  err = nomp_finalize();
  nomp_test_chk(err);

  return 0;
}

// NOMP_BACKEND value is not case sensitive.
static int test_backend_case_insensitivity(int argc, const char **argv) {
  setenv("NOMP_BACKEND", "oPenCl", 1);
  int err = nomp_init(argc, argv);
  unsetenv("NOMP_BACKEND");

  nomp_test_chk(err);

  err = nomp_finalize();
  nomp_test_chk(err);

  return 0;
}

// Run with a valid NOMP_PLATFORM_ID environment variable.
static int test_valid_platform_id(int argc, const char **argv) {
  setenv("NOMP_PLATFORM_ID", "0", 1);
  int err = nomp_init(argc, argv);
  unsetenv("NOMP_PLATFORM_ID");

  nomp_test_chk(err);

  err = nomp_finalize();
  nomp_test_chk(err);

  return 0;
}

// Run with a valid NOMP_DEVICE_ID  environment variable.
static int test_valid_device_id(int argc, const char **argv) {
  setenv("NOMP_DEVICE_ID", "0", 1);
  int err = nomp_init(argc, argv);
  unsetenv("NOMP_DEVICE_ID");

  nomp_test_chk(err);

  err = nomp_finalize();
  nomp_test_chk(err);

  return 0;
}

int main(int argc, const char *argv[]) {
  int err = 0, argsc = 6;
  const char *args0[] = {"-b", "opencl", "-d", "0", "-p", "0"};
  err |= SUBTEST(test_invalid_backend, argsc, args0);
  err |= SUBTEST(test_invalid_platform_id, argsc, args0);
  err |= SUBTEST(test_invalid_device_id, argsc, args0);

  const char *args1[] = {"-b", "invalid", "-d", "0", "-p", "0"};
  err |= SUBTEST(test_valid_backend, argsc, args1);
  err |= SUBTEST(test_backend_case_insensitivity, argsc, args1);

  char *max_int = tcalloc(char, 100);
  snprintf(max_int, 100, "%d", INT_MAX);

  const char *args2[] = {"-b", "opencl", "-d", "0", "-p", max_int};
  err |= SUBTEST(test_valid_platform_id, argsc, args2);

  const char *args3[] = {"-b", "opencl", "-d", max_int, "-p", "0"};
  err |= SUBTEST(test_valid_device_id, argsc, args3);

  tfree(max_int);

  return err;
}
