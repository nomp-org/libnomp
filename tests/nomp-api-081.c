#include "nomp-test.h"
#include "nomp.h"
#include <limits.h>

int main(int argc, const char *argv[]) {

  int length = snprintf(NULL, 0, "%d", INT_MAX);
  char *int_max_str = tcalloc(char, length + 1);
  snprintf(int_max_str, length + 1, "%d", INT_MAX);

  // Start with initially passed values
  int err = nomp_init(argc, argv);
  nomp_chk(err);
  err = nomp_finalize();

  // Set environment variable with invalid backend
  setenv("NOMP_BACKEND", "invalid", 1);
  err = nomp_init(argc, argv);
  nomp_assert(nomp_get_log_no(err) == NOMP_USER_INPUT_IS_INVALID);
  err = nomp_finalize();
  nomp_assert(nomp_get_log_no(err) == NOMP_RUNTIME_NOT_INITIALIZED);

  // Setting environment variable with valid backend
  setenv("NOMP_BACKEND", "opencl", 1);
  err = nomp_init(argc, argv);
  nomp_chk(err);
  err = nomp_finalize();
  nomp_chk(err);

  // // Environment variable has higher priority
  const char *args1[] = {" ", "-b", "invalid", "-d", "0", "-p", "0"};
  int argsc = 7;
  err = nomp_init(argsc, args1);
  nomp_chk(err);
  err = nomp_finalize();
  nomp_chk(err);

  // Environment variable case does not matter
  setenv("NOMP_BACKEND", "oPenCl", 1);
  err = nomp_init(argc, argv);
  nomp_chk(err);
  err = nomp_finalize();
  nomp_chk(err);
  unsetenv("NOMP_BACKEND");

  // For invalid platform-id environment variable, passed value is used
  setenv("NOMP_PLATFORM_ID", "invalid", 1);
  err = nomp_init(argc, argv);
  nomp_chk(err);
  err = nomp_finalize();
  nomp_chk(err);

  // If both are invalid should return an error
  const char *args2[] = {" ", "-b", "opencl", "-d", "0", "-p", "-1"};
  argsc = 7;
  err = nomp_init(argsc, args2);

  nomp_assert(nomp_get_log_no(err) == NOMP_USER_PLATFORM_IS_INVALID);
  err = nomp_finalize();
  nomp_assert(nomp_get_log_no(err) == NOMP_RUNTIME_NOT_INITIALIZED);

  // If platform-id environment variable is positive, it should have higher
  // priority.
  setenv("NOMP_PLATFORM_ID", int_max_str, 1);
  err = nomp_init(argc, argv);
  nomp_assert(nomp_get_log_no(err) == NOMP_USER_PLATFORM_IS_INVALID);
  err = nomp_finalize();
  nomp_assert(nomp_get_log_no(err) == NOMP_RUNTIME_NOT_INITIALIZED);

  // Run with a valid platform-id environment variable
  setenv("NOMP_PLATFORM_ID", "0", 1);
  const char *args3[] = {" ", "-b", "opencl", "-d", "0", "-p", "-1"};
  argsc = 7;
  err = nomp_init(argsc, args3);
  nomp_chk(err);
  err = nomp_finalize();
  nomp_chk(err);
  unsetenv("NOMP_PLATFORM_ID");

  // For invalid device-id environment variable, passed value is used
  setenv("NOMP_DEVICE_ID", "invalid", 1);
  err = nomp_init(argc, argv);
  nomp_chk(err);
  err = nomp_finalize();
  nomp_chk(err);

  // If both are invalid should return an error
  const char *args4[] = {" ", "-b", "opencl", "-d", "-1", "-p", "0"};
  argsc = 7;
  err = nomp_init(argsc, args4);
  nomp_assert(nomp_get_log_no(err) == NOMP_USER_DEVICE_IS_INVALID);
  err = nomp_finalize();
  nomp_assert(nomp_get_log_no(err) == NOMP_RUNTIME_NOT_INITIALIZED);

  // If device-id environment variable is positive, it should have higher
  // priority.
  setenv("NOMP_DEVICE_ID", int_max_str, 1);
  err = nomp_init(argc, argv);
  nomp_assert(nomp_get_log_no(err) == NOMP_USER_DEVICE_IS_INVALID);
  err = nomp_finalize();
  nomp_assert(nomp_get_log_no(err) == NOMP_RUNTIME_NOT_INITIALIZED);

  // Run with a valid device-id environment variable
  setenv("NOMP_DEVICE_ID", "0", 1);
  const char *args5[] = {" ", "-b", "opencl", "-d", "-1", "-p", "0"};
  argsc = 7;
  err = nomp_init(argsc, args5);
  nomp_chk(err);
  err = nomp_finalize();
  nomp_chk(err);
  unsetenv("NOMP_DEVICE_ID");

  tfree(int_max_str);
  return 0;
}