#include "nomp-test.h"
#include "nomp.h"
#include <limits.h>

int main(int argc, char *argv[]) {
  char *backend;
  int device, platform;
  parse_input(argc, argv, &backend, &device, &platform);

  int length = snprintf(NULL, 0, "%d", INT_MAX);
  char *int_max_str = tcalloc(char, length + 1);
  snprintf(int_max_str, length + 1, "%d", INT_MAX);

  // Start with initially passed values
  int err = nomp_init(backend, platform, device);
  nomp_chk(err);
  err = nomp_finalize();
  nomp_chk(err);

  // Set environment variable with invalid backend
  setenv("NOMP_BACKEND", "invalid", 1);
  err = nomp_init(backend, platform, device);
  nomp_assert(nomp_get_log_no(err) == NOMP_USER_INPUT_IS_INVALID);
  err = nomp_finalize();
  nomp_assert(nomp_get_log_no(err) == NOMP_RUNTIME_NOT_INITIALIZED);

  // Setting environment variable with valid backend
  setenv("NOMP_BACKEND", "opencl", 1);
  err = nomp_init(backend, platform, device);
  nomp_chk(err);
  err = nomp_finalize();
  nomp_chk(err);

  // // Environment variable has higher priority
  err = nomp_init("invalid", platform, device);
  nomp_chk(err);
  err = nomp_finalize();
  nomp_chk(err);

  // Environment variable case does not matter
  setenv("NOMP_BACKEND", "oPenCl", 1);
  err = nomp_init(backend, platform, device);
  nomp_chk(err);
  err = nomp_finalize();
  nomp_chk(err);
  unsetenv("NOMP_BACKEND");

  // For invalid platform-id environment variable, passed value is used
  setenv("NOMP_PLATFORM_ID", "invalid", 1);
  err = nomp_init(backend, platform, device);
  nomp_chk(err);
  err = nomp_finalize();
  nomp_chk(err);

  // If both are invalid should return an error
  err = nomp_init(backend, INT_MAX, device);
  nomp_assert(nomp_get_log_no(err) == NOMP_USER_PLATFORM_IS_INVALID);
  err = nomp_finalize();
  nomp_assert(nomp_get_log_no(err) == NOMP_RUNTIME_NOT_INITIALIZED);

  // If platform-id environment variable is positive, it should have higher
  // priority.
  setenv("NOMP_PLATFORM_ID", int_max_str, 1);
  err = nomp_init(backend, platform, device);
  nomp_assert(nomp_get_log_no(err) == NOMP_USER_PLATFORM_IS_INVALID);
  err = nomp_finalize();
  nomp_assert(nomp_get_log_no(err) == NOMP_RUNTIME_NOT_INITIALIZED);

  // Run with a valid platform-id environment variable
  setenv("NOMP_PLATFORM_ID", "0", 1);
  err = nomp_init(backend, INT_MAX, device);
  nomp_chk(err);
  err = nomp_finalize();
  nomp_chk(err);
  unsetenv("NOMP_PLATFORM_ID");

  // For invalid device-id environment variable, passed value is used
  setenv("NOMP_DEVICE_ID", "invalid", 1);
  err = nomp_init(backend, platform, device);
  nomp_chk(err);
  err = nomp_finalize();
  nomp_chk(err);

  // If both are invalid should return an error
  err = nomp_init(backend, platform, INT_MAX);
  nomp_assert(nomp_get_log_no(err) == NOMP_USER_DEVICE_IS_INVALID);
  err = nomp_finalize();
  nomp_assert(nomp_get_log_no(err) == NOMP_RUNTIME_NOT_INITIALIZED);

  // If device-id environment variable is positive, it should have higher
  // priority.
  setenv("NOMP_DEVICE_ID", int_max_str, 1);
  err = nomp_init(backend, platform, device);
  nomp_assert(nomp_get_log_no(err) == NOMP_USER_DEVICE_IS_INVALID);
  err = nomp_finalize();
  nomp_assert(nomp_get_log_no(err) == NOMP_RUNTIME_NOT_INITIALIZED);

  // Run with a valid device-id environment variable
  setenv("NOMP_DEVICE_ID", "0", 1);
  err = nomp_init(backend, platform, INT_MAX);
  nomp_chk(err);
  err = nomp_finalize();
  nomp_chk(err);
  unsetenv("NOMP_DEVICE_ID");

  tfree(int_max_str);
  return 0;
}
