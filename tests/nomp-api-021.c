#include "nomp-test.h"
#include <limits.h>

#define set_test_env(VAR, ENVVAR, ENVVAL)                                      \
  {                                                                            \
    VAR = nomp_copy_env(ENVVAR, NOMP_TEST_MAX_BUFFER_SIZE);                    \
    setenv(ENVVAR, ENVVAL, 1);                                                 \
  }

#define reset_env(VAR, ENVVAR)                                                 \
  {                                                                            \
    if (VAR) setenv(ENVVAR, VAR, 1);                                           \
    else unsetenv(ENVVAR);                                                     \
    nomp_free(&VAR);                                                           \
  }

// NOMP_BACKEND environment variable with invalid value.
static int test_invalid_nomp_backend(int argc, const char **argv) {
  char *backend = NULL;
  set_test_env(backend, "NOMP_BACKEND", "invalid");

  int err = nomp_init(&argc, &argv);
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

  int err = nomp_init(&argc, &argv);
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

  int err = nomp_init(&argc, &argv);
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

  nomp_test_check(nomp_init(&argc, &argv));
  nomp_test_check(nomp_finalize_excluding_interpreter());

  reset_env(platform, "NOMP_PLATFORM");

  return 0;
}

// Run with a valid NOMP_DEVICE  environment variable.
static int test_valid_device_id(int argc, const char **argv) {
  char *device = NULL;
  set_test_env(device, "NOMP_DEVICE", "0");

  nomp_test_check(nomp_init(&argc, &argv));
  nomp_test_check(nomp_finalize());

  reset_env(device, "NOMP_DEVICE");

  return 0;
}

#define SUBTEST_WITH_FRESH_ARGS(FN, SKIP)                                      \
  {                                                                            \
    argcn = copy_args(argvn, pristine, argc0, SKIP);                           \
    err |= SUBTEST(FN, argcn, (const char **)argvn);                           \
    free_args(argvn, argcn);                                                   \
  }

// Build a fresh, writable copy of the pristine command line arguments. Since
// nomp_init() rewrites the argument array in place, every call needs its own.
static unsigned copy_args(char **dst, char *const *src, unsigned n,
                          const char *skip) {
  unsigned m = 0;
  for (unsigned i = 0; i < n; i++) {
    if (skip && !strncmp(src[i], skip, NOMP_TEST_MAX_BUFFER_SIZE)) continue;
    dst[m++] = strndup(src[i], NOMP_TEST_MAX_BUFFER_SIZE);
  }
  return m;
}

static void free_args(char **a, unsigned n) {
  for (unsigned i = 0; i < n; i++)
    nomp_free(&a[i]);
}

int main(int argc, const char *argv[]) {
  nomp_test_assert(argc >= 0 && argc <= 64);

  // Snapshot the original arguments: nomp_init() modifies its argv in place.
  char    *pristine[64];
  unsigned argc0 = (unsigned)argc;
  for (unsigned i = 0; i < argc0; i++)
    pristine[i] = strndup(argv[i], NOMP_TEST_MAX_BUFFER_SIZE);

  int   err = 0;
  char *argvn[64];
  unsigned argcn;

  SUBTEST_WITH_FRESH_ARGS(test_invalid_nomp_backend, NULL);
  SUBTEST_WITH_FRESH_ARGS(test_invalid_platform_id, NULL);
  SUBTEST_WITH_FRESH_ARGS(test_invalid_device_id, NULL);
  // Everything except `--nomp-platform` / `--nomp-device`.
  SUBTEST_WITH_FRESH_ARGS(test_valid_platform_id, "--nomp-platform");
  SUBTEST_WITH_FRESH_ARGS(test_valid_device_id, "--nomp-device");

  free_args(pristine, argc0);
  return err;
}

#undef SUBTEST_WITH_FRESH_ARGS
#undef set_test_env
#undef reset_env
