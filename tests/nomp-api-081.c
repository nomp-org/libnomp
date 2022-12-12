#include "nomp-test.h"
#include "nomp.h"
#include <limits.h>

// Set environment variable with invalid backend
static int test_invalid_backed(int argc, const char **argv) {
  // Start with initially passed values
  int err = nomp_init(argc, argv);
  nomp_test_chk(err);
  err = nomp_finalize();
  nomp_test_chk(err);

  setenv("NOMP_BACKEND", "invalid", 1);
  err = nomp_init(argc, argv);
  nomp_test_assert(nomp_get_log_no(err) == NOMP_USER_INPUT_IS_INVALID);
  err = nomp_finalize();
  nomp_test_assert(nomp_get_log_no(err) == NOMP_RUNTIME_NOT_INITIALIZED);
  return 0;
}

// Setting environment variable with valid backend
static int test_valid_backend(int argc, const char **argv) {
  setenv("NOMP_BACKEND", "opencl", 1);
  int err = nomp_init(argc, argv);
  nomp_test_chk(err);
  err = nomp_finalize();
  nomp_test_chk(err);
  return 0;
}

// Environment variable has higher priority
static int test_environment_variable_priority(int argc, const char **argv) {
  int err = nomp_init(argc, argv);
  nomp_test_chk(err);
  err = nomp_finalize();
  nomp_test_chk(err);
  return 0;
}

// Environment variable case does not matter {
static int test_environment_variable_case_sensitivity(int argc,
                                                      const char **argv) {
  setenv("NOMP_BACKEND", "oPenCl", 1);
  int err = nomp_init(argc, argv);
  nomp_test_chk(err);
  err = nomp_finalize();
  nomp_test_chk(err);
  unsetenv("NOMP_BACKEND");
  return 0;
}

// For invalid platform-id environment variable, passed value is used
static int test_invalid_platform_id(int argc, const char **argv) {
  setenv("NOMP_PLATFORM_ID", "invalid", 1);
  int err = nomp_init(argc, argv);
  nomp_test_chk(err);
  err = nomp_finalize();
  nomp_test_chk(err);
  return 0;
}

// If both are invalid should return an error
static int test_invalid_both_platform_id(int argc, const char **argv) {
  int err = nomp_init(argc, argv);
  nomp_test_assert(nomp_get_log_no(err) == NOMP_USER_PLATFORM_IS_INVALID);
  err = nomp_finalize();
  nomp_test_assert(nomp_get_log_no(err) == NOMP_RUNTIME_NOT_INITIALIZED);
  return 0;
}

// If platform-id environment variable is positive, it should have higher
// priority.
static int test_positive_platform_id(int argc, const char **argv,
                                     char *int_max_str) {
  setenv("NOMP_PLATFORM_ID", int_max_str, 1);
  int err = nomp_init(argc, argv);
  nomp_test_assert(nomp_get_log_no(err) == NOMP_USER_PLATFORM_IS_INVALID);
  err = nomp_finalize();
  nomp_test_assert(nomp_get_log_no(err) == NOMP_RUNTIME_NOT_INITIALIZED);
  return 0;
}

// Run with a valid platform-id environment variable
static int test_valid_platform_id(int argc, const char **argv) {
  setenv("NOMP_PLATFORM_ID", "0", 1);
  int err = nomp_init(argc, argv);
  nomp_test_chk(err);
  err = nomp_finalize();
  nomp_test_chk(err);
  unsetenv("NOMP_PLATFORM_ID");
  return 0;
}

// For invalid device-id environment variable, passed value is used
static int test_invalid_device_id(int argc, const char **argv) {
  setenv("NOMP_DEVICE_ID", "invalid", 1);
  int err = nomp_init(argc, argv);
  nomp_test_chk(err);
  err = nomp_finalize();
  nomp_test_chk(err);
  return 0;
}

// If both are invalid should return an error
static int test_invalid_both_device_id(int argc, const char **argv) {
  int err = nomp_init(argc, argv);
  nomp_test_assert(nomp_get_log_no(err) == NOMP_USER_DEVICE_IS_INVALID);
  err = nomp_finalize();
  nomp_test_assert(nomp_get_log_no(err) == NOMP_RUNTIME_NOT_INITIALIZED);
  return 0;
}

// If device-id environment variable is positive, it should have higher
// priority.
static int test_positive_device_id(int argc, const char **argv,
                                   char *int_max_str) {
  setenv("NOMP_DEVICE_ID", int_max_str, 1);
  int err = nomp_init(argc, argv);
  nomp_test_assert(nomp_get_log_no(err) == NOMP_USER_DEVICE_IS_INVALID);
  err = nomp_finalize();
  nomp_test_assert(nomp_get_log_no(err) == NOMP_RUNTIME_NOT_INITIALIZED);
  return 0;
}

// Run with a valid device-id environment variable
static int test_valid_device_id(int argc, const char **argv) {
  setenv("NOMP_DEVICE_ID", "0", 1);
  int err = nomp_init(argc, argv);
  nomp_test_chk(err);
  err = nomp_finalize();
  nomp_test_chk(err);
  unsetenv("NOMP_DEVICE_ID");
  return 0;
}

int main(int argc, const char *argv[]) {

  int length = snprintf(NULL, 0, "%d", INT_MAX);
  char *int_max_str = tcalloc(char, length + 1);
  snprintf(int_max_str, length + 1, "%d", INT_MAX);
  int err = 0;

  err |= SUBTEST(test_invalid_backed, argc, argv);
  err |= SUBTEST(test_valid_backend, argc, argv);

  const char *args1[] = {"-b", "invalid", "-d", "0", "-p", "0"};
  int argsc = 6;
  err |= SUBTEST(test_environment_variable_priority, argsc, args1);
  err |= SUBTEST(test_environment_variable_case_sensitivity, argc, argv);
  err |= SUBTEST(test_invalid_platform_id, argc, argv);

  const char *args2[] = {"-b", "opencl", "-d", "0", "-p", int_max_str};
  argsc = 6;
  err |= SUBTEST(test_invalid_both_platform_id, argsc, args2);
  err |= SUBTEST(test_positive_platform_id, argc, argv, int_max_str);

  const char *args3[] = {"-b", "opencl", "-d", "0", "-p", int_max_str};
  argsc = 6;
  err |= SUBTEST(test_valid_platform_id, argsc, args3);
  err |= SUBTEST(test_invalid_device_id, argc, argv);

  const char *args4[] = {"-b", "opencl", "-d", int_max_str, "-p", "0"};
  argsc = 6;
  err |= SUBTEST(test_invalid_both_device_id, argsc, args4);
  err |= SUBTEST(test_positive_device_id, argc, argv, int_max_str);

  const char *args5[] = {"-b", "opencl", "-d", "-1", "-p", "0"};
  argsc = 6;
  err |= SUBTEST(test_valid_device_id, argsc, args5);

  tfree(int_max_str);
  return err;
}