#include "nomp-test.h"
#include "nomp.h"
#include <limits.h>

// Environment variable with invalid backend should be an error.
static int test_invalid_backend(int argc, const char *argv[]) {
  setenv("NOMP_BACKEND", "invalid", 1);
  int err = nomp_init(argc, argv);
  nomp_test_assert(nomp_get_log_no(err) == NOMP_USER_INPUT_IS_INVALID);

  err = nomp_finalize();
  nomp_test_assert(nomp_get_log_no(err) == NOMP_RUNTIME_NOT_INITIALIZED);

  unsetenv("NOMP_BACKEND");
  return 0;
}

// Environment variable with valid backend should succeed.
static int test_valid_backend(int argc, const char **argv) {
  setenv("NOMP_BACKEND", "opencl", 1);
  int err = nomp_init(argc, argv);
  nomp_test_chk(err);

  err = nomp_finalize();
  nomp_test_chk(err);

  unsetenv("NOMP_BACKEND");
  return 0;
}

// Environment variable has higher priority, so it should override options
// passed to nomp_init().
static int test_env_var_priority() {
  setenv("NOMP_BACKEND", "opencl", 1);
  const char *argv[] = {"-b", "invalid", "-d", "0", "-p", "0"};
  int argc = 6;
  int err = nomp_init(argc, argv);
  nomp_test_chk(err);

  err = nomp_finalize();
  nomp_test_chk(err);

  unsetenv("NOMP_BACKEND");
  return 0;
}

// Environment variable case shouldn't matter.
static int test_env_var_case_sensitivity(int argc, const char **argv) {
  setenv("NOMP_BACKEND", "oPenCl", 1);
  int err = nomp_init(argc, argv);
  nomp_test_chk(err);

  err = nomp_finalize();
  nomp_test_chk(err);

  unsetenv("NOMP_BACKEND");
  return 0;
}

// If platform id environment variable is invalid, value passed in the options
// is used.
static int test_invalid_platform_id_env_var(int argc, const char **argv) {
  setenv("NOMP_PLATFORM_ID", "invalid", 1);
  int err = nomp_init(argc, argv);
  nomp_test_chk(err);

  err = nomp_finalize();
  nomp_test_chk(err);

  unsetenv("NOMP_PLATFORM_ID");
  return 0;
}

// If platform-id environment variable is positive, it should have higher
// priority. FIXME: This test is broken due to bugs in error API.
static int test_positive_platform_id_env_var(int argc, const char **argv) {
  setenv("NOMP_PLATFORM_ID", "1e20", 1);
  int err = nomp_init(argc, argv);
  nomp_test_assert(nomp_get_log_no(err) == NOMP_USER_PLATFORM_IS_INVALID);

  err = nomp_finalize();
  nomp_test_assert(nomp_get_log_no(err) == NOMP_RUNTIME_NOT_INITIALIZED);

  unsetenv("NOMP_PLATFORM_ID");
  return 0;
}

// If both command line option and env. var values are invalid, default value
// should be used.
static int test_invalid_platform_id() {
  setenv("NOMP_PLATFORM_ID", "invalid", 1);
  const char *argv[] = {"-b", "opencl", "-d", "0", "-p", "-1"};
  int argc = 6;
  int err = nomp_init(argc, argv);
  nomp_test_chk(err);

  err = nomp_finalize();
  nomp_test_chk(err);

  unsetenv("NOMP_PLATFORM_ID");
  return 0;
}

// If the device id env var is invalid, passed value is used.
static int test_invalid_device_id_env_var(int argc, const char **argv) {
  setenv("NOMP_DEVICE_ID", "invalid", 1);
  int err = nomp_init(argc, argv);
  nomp_test_chk(err);

  err = nomp_finalize();
  nomp_test_chk(err);

  unsetenv("NOMP_DEVICE_ID");
  return 0;
}

// If device id environment variable is positive, it should have higher
// priority. FIXME: This test is broken due to bugs in error API.
static int test_positive_device_id_env_var(int argc, const char **argv) {
  setenv("NOMP_DEVICE_ID", "1e20", 1);
  int err = nomp_init(argc, argv);
  nomp_test_assert(nomp_get_log_no(err) == NOMP_USER_DEVICE_IS_INVALID);

  err = nomp_finalize();
  nomp_test_assert(nomp_get_log_no(err) == NOMP_RUNTIME_NOT_INITIALIZED);

  unsetenv("NOMP_DEVICE_ID");
  return 0;
}

// If both env var and option are invalid, default value should be used.
static int test_invalid_device_id() {
  setenv("NOMP_DEVICE_ID", "invalid", 1);
  const char *argv[] = {"-b", "opencl", "-d", "-1", "-p", "0"};
  int argc = 6;
  int err = nomp_init(argc, argv);
  nomp_test_chk(err);

  err = nomp_finalize();
  nomp_test_chk(err);

  unsetenv("NOMP_DEVICE_ID");
  return 0;
}

int main(int argc, const char *argv[]) {
  int err = SUBTEST(test_invalid_backend, argc, argv);
  err |= SUBTEST(test_valid_backend, argc, argv);

  err |= SUBTEST(test_env_var_priority);
  err |= SUBTEST(test_env_var_case_sensitivity, argc, argv);

  err |= SUBTEST(test_invalid_platform_id_env_var, argc, argv);
  // err |= SUBTEST(test_positive_platform_id_env_var, argc, argv);
  err |= SUBTEST(test_invalid_platform_id);

  err |= SUBTEST(test_invalid_device_id_env_var, argc, argv);
  // err |= SUBTEST(test_positive_device_id_env_var, argc, argv);
  err |= SUBTEST(test_invalid_device_id);

  return err;
}
