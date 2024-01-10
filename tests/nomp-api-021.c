#include "nomp-test.h"
#include <limits.h>

#define set_test_env(VAR, ENVVAR, ENVVAL)                                      \
  {                                                                            \
    VAR = nomp_copy_env(ENVVAR, NOMP_TEST_MAX_BUFFER_SIZE);                    \
    setenv(ENVVAR, ENVVAL, 1);                                                 \
  }

#define reset_env(VAR, ENVVAR)                                                 \
  {                                                                            \
    if (VAR)                                                                   \
      setenv(ENVVAR, VAR, 1);                                                  \
    else                                                                       \
      unsetenv(ENVVAR);                                                        \
    nomp_free(&VAR);                                                           \
  }

// NOMP_BACKEND environment variable with invalid value.
static int test_invalid_nomp_backend(int argc, const char **argv) {
  char *backend = NULL;
  set_test_env(backend, "NOMP_BACKEND", "invalid");

  int err = nomp_init(argc, argv);
  nomp_test_assert(nomp_get_err_no(err) == NOMP_USER_INPUT_IS_INVALID);

  err = nomp_finalize();
  nomp_test_assert(err == NOMP_FINALIZE_FAILURE);
  nomp_test_assert(nomp_get_err_no(err) == NOMP_USER_LOG_ID_IS_INVALID);

  reset_env(backend, "NOMP_BACKEND");

  return 0;
}

// NOMP_PLATFORM environment variable with invalid value.
static int test_invalid_platform_id(int argc, const char **argv) {
  char *platform = NULL;
  set_test_env(platform, "NOMP_PLATFORM", "invalid");

  int err = nomp_init(argc, argv);
  nomp_test_assert(nomp_get_err_no(err) == NOMP_USER_INPUT_IS_INVALID);

  err = nomp_finalize();
  nomp_test_assert(err == NOMP_FINALIZE_FAILURE);
  nomp_test_assert(nomp_get_err_no(err) == NOMP_USER_LOG_ID_IS_INVALID);

  reset_env(platform, "NOMP_PLATFORM");

  return 0;
}

// NOMP_DEVICE environment variable with invalid value.
static int test_invalid_device_id(int argc, const char **argv) {
  char *device = NULL;
  set_test_env(device, "NOMP_DEVICE", "invalid");

  int err = nomp_init(argc, argv);
  nomp_test_assert(nomp_get_err_no(err) == NOMP_USER_INPUT_IS_INVALID);

  err = nomp_finalize();
  nomp_test_assert(err == NOMP_FINALIZE_FAILURE);
  nomp_test_assert(nomp_get_err_no(err) == NOMP_USER_LOG_ID_IS_INVALID);

  reset_env(device, "NOMP_DEVICE");

  return 0;
}

// Run with a valid NOMP_PLATFORM environment variable.
static int test_valid_platform_id(int argc, const char **argv) {
  char *platform = NULL;
  set_test_env(platform, "NOMP_PLATFORM", "0");

  nomp_test_check(nomp_init(argc, argv));
  nomp_test_check(nomp_finalize());

  reset_env(platform, "NOMP_PLATFORM");

  return 0;
}

// Run with a valid NOMP_DEVICE  environment variable.
static int test_valid_device_id(int argc, const char **argv) {
  char *device = NULL;
  set_test_env(device, "NOMP_DEVICE", "0");

  nomp_test_check(nomp_init(argc, argv));
  nomp_test_check(nomp_finalize());

  reset_env(device, "NOMP_DEVICE");

  return 0;
}

int main(int argc, const char *argv[]) {
  int err = 0;
  err |= SUBTEST(test_invalid_nomp_backend, argc, argv);
  err |= SUBTEST(test_invalid_platform_id, argc, argv);
  err |= SUBTEST(test_invalid_device_id, argc, argv);

  nomp_test_assert(argc <= 64);
  char *argvn[64];

  // Copy everything except `--nomp-platform` to new command line args.
  unsigned argcn = 0;
  for (unsigned i = 0; i < (unsigned)argc; i++) {
    if (strncmp(argv[i], "--nomp-platform", NOMP_TEST_MAX_BUFFER_SIZE))
      argvn[argcn] = strndup(argv[i], NOMP_TEST_MAX_BUFFER_SIZE), argcn++;
  }
  err |= SUBTEST(test_valid_platform_id, argcn, (const char **)argvn);
  for (unsigned i = 0; i < argcn; i++)
    nomp_free(&argvn[i]);

  // Copy everything except `--nomp-device` to new command line args.
  argcn = 0;
  for (unsigned i = 0; i < (unsigned)argc; i++) {
    if (strncmp(argv[i], "--nomp-device", NOMP_TEST_MAX_BUFFER_SIZE))
      argvn[argcn] = strndup(argv[i], NOMP_TEST_MAX_BUFFER_SIZE), argcn++;
  }
  // FIXME: This fails due to multiple imports of annotation script.
  // This test re initializes the library, which causes the annotation script
  // to be imported again, which causes the error. This seems to be an
  // issue with PyImport_Import().
  // err |= SUBTEST(test_valid_device_id, argcn, (const char **)argvn);
  for (unsigned i = 0; i < argcn; i++)
    nomp_free(&argvn[i]);

  return err;
}

#undef set_test_env
#undef reset_env
