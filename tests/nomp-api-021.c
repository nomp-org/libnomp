#include "nomp-test.h"
#include <limits.h>

#define set_test_env(VAR, ENVVAR, ENVVAL)                                      \
  {                                                                            \
    VAR = nomp_copy_env(ENVVAR, NOMP_TEST_MAX_BUFSIZ);                         \
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
  nomp_test_assert(nomp_get_log_no(err) == NOMP_USER_INPUT_IS_INVALID);

  nomp_test_assert(nomp_get_log_no(nomp_finalize()) == NOMP_FINALIZE_FAILURE);

  reset_env(backend, "NOMP_BACKEND");

  return 0;
}

// NOMP_PLATFORM environment variable with invalid value.
static int test_invalid_platform_id(int argc, const char **argv) {
  char *platform = NULL;
  set_test_env(platform, "NOMP_PLATFORM", "invalid");

  int err = nomp_init(argc, argv);
  nomp_test_assert(nomp_get_log_no(err) == NOMP_USER_INPUT_IS_INVALID);

  nomp_test_assert(nomp_get_log_no(nomp_finalize()) == NOMP_FINALIZE_FAILURE);

  reset_env(platform, "NOMP_PLATFORM");

  return 0;
}

// NOMP_DEVICE environment variable with invalid value.
static int test_invalid_device_id(int argc, const char **argv) {
  char *device = NULL;
  set_test_env(device, "NOMP_DEVICE", "invalid");

  int err = nomp_init(argc, argv);
  nomp_test_assert(nomp_get_log_no(err) == NOMP_USER_INPUT_IS_INVALID);

  nomp_test_assert(nomp_get_log_no(nomp_finalize()) == NOMP_FINALIZE_FAILURE);

  reset_env(device, "NOMP_DEVICE");

  return 0;
}

// NOMP_VERBOSE environment variable with invalid value.
static int test_invalid_nomp_verbose(int argc, const char **argv) {
  char *verbose = NULL;
  set_test_env(verbose, "NOMP_VERBOSE", "4");

  int err = nomp_init(argc, argv);
  nomp_test_assert(nomp_get_log_no(err) == NOMP_USER_INPUT_IS_INVALID);

  char *desc = nomp_get_log_str(err);
  int eq = logcmp(
      desc, "\\[Error\\] .*libnomp\\/src\\/log.c:[0-9]* Invalid verbose level "
            "4 is provided. The value should be within the range 0-3.");
  nomp_test_assert(eq);
  nomp_free(&desc);

  nomp_test_assert(nomp_get_log_no(nomp_finalize()) == NOMP_FINALIZE_FAILURE);

  reset_env(verbose, "NOMP_VERBOSE");

  return 0;
}

// Run with a valid NOMP_PLATFORM environment variable.
static int test_valid_platform_id(int argc, const char **argv) {
  char *platform = NULL;
  set_test_env(platform, "NOMP_PLATFORM", "0");

  nomp_test_chk(nomp_init(argc, argv));
  nomp_test_chk(nomp_finalize());

  reset_env(platform, "NOMP_PLATFORM");

  return 0;
}

// Run with a valid NOMP_DEVICE  environment variable.
static int test_valid_device_id(int argc, const char **argv) {
  char *device = NULL;
  set_test_env(device, "NOMP_DEVICE", "0");

  nomp_test_chk(nomp_init(argc, argv));
  nomp_test_chk(nomp_finalize());

  reset_env(device, "NOMP_DEVICE");

  return 0;
}

int main(int argc, const char *argv[]) {
  int err = 0;
  err |= SUBTEST(test_invalid_nomp_backend, argc, argv);
  err |= SUBTEST(test_invalid_platform_id, argc, argv);
  err |= SUBTEST(test_invalid_device_id, argc, argv);
  err |= SUBTEST(test_invalid_nomp_verbose, argc, argv);

  nomp_test_assert(argc <= 64);
  char *argvn[64];

  // Copy everything but `--nomp-platform` to new command line args.
  int argcn = 0;
  for (unsigned i = 0; i < argc; i++) {
    if (strncmp(argv[i], "--nomp-platform", NOMP_TEST_MAX_BUFSIZ))
      argvn[argcn] = strndup(argv[i], NOMP_TEST_MAX_BUFSIZ), argcn++;
  }
  err |= SUBTEST(test_valid_platform_id, argcn, (const char **)argvn);
  for (unsigned i = 0; i < argcn; i++)
    nomp_free(&argvn[i]);

  // Copy everything but `--nomp-device` to new command line args.
  argcn = 0;
  for (unsigned i = 0; i < argc; i++) {
    if (strncmp(argv[i], "--nomp-device", NOMP_TEST_MAX_BUFSIZ))
      argvn[argcn] = strndup(argv[i], NOMP_TEST_MAX_BUFSIZ), argcn++;
  }
  err |= SUBTEST(test_valid_device_id, argcn, (const char **)argvn);
  for (unsigned i = 0; i < argcn; i++)
    nomp_free(&argvn[i]);

  return err;
}
