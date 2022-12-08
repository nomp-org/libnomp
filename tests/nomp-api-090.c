#include "nomp-test.h"
#include "nomp.h"
#include <limits.h>

// Set environment variable with invalid backend
static int test_invalid_backed(char *backend, int platform, int device) {
  // Start with initially passed values
  int err = nomp_init(backend, platform, device);
  nomp_test_chk(err);
  err = nomp_finalize();
  nomp_test_chk(err);

  setenv("NOMP_BACKEND", "invalid", 1);
  err = nomp_init(backend, platform, device);
  nomp_test_assert(nomp_get_log_no(err) == NOMP_USER_INPUT_IS_INVALID);
  err = nomp_finalize();
  nomp_test_assert(nomp_get_log_no(err) == NOMP_RUNTIME_NOT_INITIALIZED);
  return 0;
}

// Setting environment variable with valid backend
static int test_valid_backend(char *backend, int platform, int device) {
  setenv("NOMP_BACKEND", "opencl", 1);
  int err = nomp_init(backend, platform, device);
  nomp_test_chk(err);
  err = nomp_finalize();
  nomp_test_chk(err);
  return 0;
}

// Environment variable has higher priority
static int test_environment_variable_priority(int platform, int device) {
  int err = nomp_init("invalid", platform, device);
  nomp_test_chk(err);
  err = nomp_finalize();
  nomp_test_chk(err);
  return 0;
}

// Environment variable case does not matter {
static int test_environment_variable_case_sensitivity(char *backend,
                                                      int platform,
                                                      int device) {
  setenv("NOMP_BACKEND", "oPenCl", 1);
  int err = nomp_init(backend, platform, device);
  nomp_test_chk(err);
  err = nomp_finalize();
  nomp_test_chk(err);
  unsetenv("NOMP_BACKEND");
  return 0;
}

// For invalid platform-id environment variable, passed value is used
static int test_invalid_platform_id(char *backend, int platform, int device) {
  setenv("NOMP_PLATFORM_ID", "invalid", 1);
  int err = nomp_init(backend, platform, device);
  nomp_test_chk(err);
  err = nomp_finalize();
  nomp_test_chk(err);
  return 0;
}

// If both are invalid should return an error
static int test_invalid_both_platform_id(char *backend, int platform,
                                         int device) {
  int err = nomp_init(backend, INT_MAX, device);
  nomp_test_assert(nomp_get_log_no(err) == NOMP_USER_PLATFORM_IS_INVALID);
  err = nomp_finalize();
  nomp_test_assert(nomp_get_log_no(err) == NOMP_RUNTIME_NOT_INITIALIZED);
  return 0;
}

// If platform-id environment variable is positive, it should have higher
// priority.
static int test_positive_platform_id(char *backend, int platform, int device,
                                     char *int_max_str) {
  setenv("NOMP_PLATFORM_ID", int_max_str, 1);
  int err = nomp_init(backend, platform, device);
  nomp_test_assert(nomp_get_log_no(err) == NOMP_USER_PLATFORM_IS_INVALID);
  err = nomp_finalize();
  nomp_test_assert(nomp_get_log_no(err) == NOMP_RUNTIME_NOT_INITIALIZED);
  return 0;
}

// Run with a valid platform-id environment variable
static int test_valid_platform_id(char *backend, int platform, int device) {
  setenv("NOMP_PLATFORM_ID", "0", 1);
  int err = nomp_init(backend, INT_MAX, device);
  nomp_test_chk(err);
  err = nomp_finalize();
  nomp_test_chk(err);
  unsetenv("NOMP_PLATFORM_ID");
  return 0;
}

// For invalid device-id environment variable, passed value is used
static int test_invalid_device_id(char *backend, int platform, int device) {
  setenv("NOMP_DEVICE_ID", "invalid", 1);
  int err = nomp_init(backend, platform, device);
  nomp_test_chk(err);
  err = nomp_finalize();
  nomp_test_chk(err);
  return 0;
}

// If both are invalid should return an error
static int test_invalid_both_device_id(char *backend, int platform,
                                       int device) {
  int err = nomp_init(backend, platform, INT_MAX);
  nomp_test_assert(nomp_get_log_no(err) == NOMP_USER_DEVICE_IS_INVALID);
  err = nomp_finalize();
  nomp_test_assert(nomp_get_log_no(err) == NOMP_RUNTIME_NOT_INITIALIZED);
  return 0;
}

// If device-id environment variable is positive, it should have higher
// priority.
static int test_positive_device_id(char *backend, int platform, int device,
                                   char *int_max_str) {
  setenv("NOMP_DEVICE_ID", int_max_str, 1);
  int err = nomp_init(backend, platform, device);
  nomp_test_assert(nomp_get_log_no(err) == NOMP_USER_DEVICE_IS_INVALID);
  err = nomp_finalize();
  nomp_test_assert(nomp_get_log_no(err) == NOMP_RUNTIME_NOT_INITIALIZED);
  return 0;
}

// Run with a valid device-id environment variable
static int test_valid_device_id(char *backend, int platform, int device) {
  setenv("NOMP_DEVICE_ID", "0", 1);
  int err = nomp_init(backend, platform, INT_MAX);
  nomp_test_chk(err);
  err = nomp_finalize();
  nomp_test_chk(err);
  unsetenv("NOMP_DEVICE_ID");
  return 0;
}

int main(int argc, char *argv[]) {
  char *backend;
  int device, platform;
  parse_input(argc, argv, &backend, &device, &platform);

  int length = snprintf(NULL, 0, "%d", INT_MAX);
  char *int_max_str = tcalloc(char, length + 1);
  snprintf(int_max_str, length + 1, "%d", INT_MAX);
  int err = 0;

  err |= SUBTEST(test_invalid_backed, backend, platform, device);
  err |= SUBTEST(test_valid_backend, backend, platform, device);
  err |= SUBTEST(test_environment_variable_priority, platform, device);
  err |= SUBTEST(test_environment_variable_case_sensitivity, backend, platform,
                 device);
  err |= SUBTEST(test_invalid_platform_id, backend, platform, device);
  err |= SUBTEST(test_invalid_both_platform_id, backend, platform, device);
  err |= SUBTEST(test_positive_platform_id, backend, platform, device,
                 int_max_str);
  err |= SUBTEST(test_valid_platform_id, backend, platform, device);
  err |= SUBTEST(test_invalid_device_id, backend, platform, device);
  err |= SUBTEST(test_invalid_both_device_id, backend, platform, device);
  err |=
      SUBTEST(test_positive_device_id, backend, platform, device, int_max_str);
  err |= SUBTEST(test_valid_device_id, backend, platform, device);

  tfree(int_max_str);
  return err;
}
