#include "nomp-test.h"

/* Sets (or unsets, if value is NULL) an environment variable and returns its
 * previous value (a heap copy, or NULL if it was unset). */
static char *test_setenv(const char *name, const char *value) {
  const char *old   = getenv(name);
  char       *saved = old ? strdup(old) : NULL;
  if (value) setenv(name, value, 1);
  else unsetenv(name);
  return saved;
}

static void test_restore_env(const char *name, char *saved) {
  if (saved) setenv(name, saved, 1);
  else unsetenv(name);
  free(saved);
}

static int test_valid_arguments(const char *backend) {
  const char  *args[7] = {"foo.c", "--nomp-backend",  backend, "--nomp-device",
                          "0",     "--nomp-platform", "0"};
  const char **argv    = args;
  int          argc    = 7;

  nomp_test_check(nomp_init(&argc, &argv));
  nomp_test_assert(argc == 1);
  nomp_test_assert(strcmp(argv[0], "foo.c") == 0);
  nomp_test_check(nomp_finalize());

  return 0;
}

static int test_ignore_non_argument_string(const char *backend) {
  const char  *args[8] = {"--nomp-backend",  backend, "cuda",
                          "--nomp-device",   "0",     "1",
                          "--nomp-platform", "0"};
  const char **argv    = args;
  int          argc    = 8;

  nomp_test_check(nomp_init(&argc, &argv));
  nomp_test_assert(argc == 2);
  nomp_test_assert(strcmp(argv[0], "cuda") == 0);
  nomp_test_assert(strcmp(argv[1], "1") == 0);
  nomp_test_check(nomp_finalize());

  return 0;
}

static int test_invalid_argument_flag(const char *backend) {
  const char  *args[8] = {"--nomp-backend",  backend, "--nomp-device", "0",
                          "--nomp-platform", "0",     "--unknown-arg", "value"};
  const char **argv    = args;
  int          argc    = 8;

  nomp_test_check(nomp_init(&argc, &argv));
  nomp_test_assert(argc == 2);
  nomp_test_assert(strcmp(argv[0], "--unknown-arg") == 0);
  nomp_test_assert(strcmp(argv[1], "value") == 0);
  nomp_test_check(nomp_finalize());

  return 0;
}

static int test_missing_argument(const char *backend) {
  const char  *args[5] = {"--nomp-backend", backend, "--nomp-device", "0",
                          "--nomp-platform"};
  const char **argv    = args;
  int          argc    = 5;
  int          err     = nomp_init(&argc, &argv);
  nomp_test_assert(nomp_get_err_no(err) == NOMP_USER_INPUT_IS_INVALID);
  nomp_test_assert(argc == 5);
  nomp_test_assert(strcmp(argv[4], "--nomp-platform") == 0);

  err = nomp_finalize();
  nomp_test_assert(err == NOMP_FINALIZE_FAILURE);
  nomp_test_assert(nomp_get_err_no(err) == NOMP_USER_LOG_ID_IS_INVALID);

  return 0;
}

static int test_unknown_option_removed(const char *backend) {
  const char *args[7] = {
      "a", "--nomp-unknown", "x", "b", "--nomp-backend", backend, "c"};
  const char **argv = args;
  int          argc = 7;

  nomp_test_check(nomp_init(&argc, &argv));
  nomp_test_assert(argc == 3);
  nomp_test_assert(strcmp(argv[0], "a") == 0);
  nomp_test_assert(strcmp(argv[1], "b") == 0);
  nomp_test_assert(strcmp(argv[2], "c") == 0);
  nomp_test_assert(argv[3] == NULL);
  nomp_test_check(nomp_finalize());

  return 0;
}

static int test_no_removal_exact_size(const char *backend) {
  // Heap array of exactly argc entries: nothing may be written, in particular
  // no terminator at argv[argc].
  const char **orig = (const char **)malloc(2 * sizeof(const char *));
  nomp_test_assert(orig != NULL);
  orig[0]           = "foo";
  orig[1]           = "bar";
  const char **argv = orig;
  int          argc = 2;

  char *saved = test_setenv("NOMP_BACKEND", backend);
  int   err   = nomp_init(&argc, &argv);
  test_restore_env("NOMP_BACKEND", saved);
  nomp_test_check(err);

  nomp_test_assert(argc == 2);
  nomp_test_assert(argv == orig);
  nomp_test_assert(strcmp(argv[0], "foo") == 0);
  nomp_test_assert(strcmp(argv[1], "bar") == 0);
  nomp_test_check(nomp_finalize());
  free(orig);

  return 0;
}

static int test_index_zero_option(const char *backend) {
  const char  *args[3] = {"--nomp-backend", backend, "prog"};
  const char **argv    = args;
  int          argc    = 3;

  nomp_test_check(nomp_init(&argc, &argv));
  nomp_test_assert(argc == 1);
  nomp_test_assert(strcmp(argv[0], "prog") == 0);
  nomp_test_check(nomp_finalize());

  return 0;
}

static int test_argc_one_not_parsed(const char *backend) {
  // A single --nomp-* token with argc == 1 is not parsed (and not removed).
  // The backend is therefore taken from the environment.
  const char  *args[1] = {"--nomp-bogus"};
  const char **argv    = args;
  int          argc    = 1;

  char *saved = test_setenv("NOMP_BACKEND", backend);
  int   err   = nomp_init(&argc, &argv);
  test_restore_env("NOMP_BACKEND", saved);
  nomp_test_check(err);

  nomp_test_assert(argc == 1);
  nomp_test_assert(strcmp(argv[0], "--nomp-bogus") == 0);
  nomp_test_check(nomp_finalize());

  return 0;
}

static int test_option_looking_value(const char *backend) {
  // The unknown option consumes "--nomp-backend" as its value; the second
  // "--nomp-backend" is the real one.
  const char  *args[5] = {"prog", "--nomp-unknown", "--nomp-backend",
                          "--nomp-backend", backend};
  const char **argv    = args;
  int          argc    = 5;

  nomp_test_check(nomp_init(&argc, &argv));
  nomp_test_assert(argc == 1);
  nomp_test_assert(strcmp(argv[0], "prog") == 0);
  nomp_test_assert(argv[1] == NULL);
  nomp_test_check(nomp_finalize());

  return 0;
}

static int test_conditional_terminator_and_stale_tail(const char *backend) {
  const char  *args[6] = {"a", "--nomp-backend", backend, "b", "c", "tail"};
  const char **argv    = args;
  int          argc    = 5;

  nomp_test_check(nomp_init(&argc, &argv));
  nomp_test_assert(argc == 3);
  nomp_test_assert(strcmp(argv[0], "a") == 0);
  nomp_test_assert(strcmp(argv[1], "b") == 0);
  nomp_test_assert(strcmp(argv[2], "c") == 0);
  // Terminator written at the new argc since something was removed.
  nomp_test_assert(argv[3] == NULL);
  // Entries strictly after the terminator are unchanged (stale).
  nomp_test_assert(strcmp(argv[4], "c") == 0);
  nomp_test_assert(strcmp(argv[5], "tail") == 0);
  nomp_test_check(nomp_finalize());

  return 0;
}

static int test_no_removal_no_terminator(const char *backend) {
  // With a spare slot present, it must stay untouched when nothing is removed.
  const char  *spare   = "spare";
  const char  *args[3] = {"foo", "bar", spare};
  const char **argv    = args;
  int          argc    = 2;

  char *saved = test_setenv("NOMP_BACKEND", backend);
  int   err   = nomp_init(&argc, &argv);
  test_restore_env("NOMP_BACKEND", saved);
  nomp_test_check(err);

  nomp_test_assert(argc == 2);
  nomp_test_assert(argv[2] == spare);
  nomp_test_check(nomp_finalize());

  return 0;
}

static int test_missing_value_unchanged(const char *backend) {
  const char *args[6] = {"prog",           "--nomp-device", "0",
                         "--nomp-backend", backend,         "--nomp-platform"};
  const char *copy[6];
  memcpy(copy, args, sizeof(args));
  const char **argv = args;
  int          argc = 6;

  int err = nomp_init(&argc, &argv);
  nomp_test_assert(nomp_get_err_no(err) == NOMP_USER_INPUT_IS_INVALID);
  nomp_test_assert(argc == 6);
  nomp_test_assert(argv == args);
  for (unsigned i = 0; i < 6; i++)
    nomp_test_assert(args[i] == copy[i]);

  return 0;
}

static int test_later_failure_leaves_compacted_argv(const char *backend) {
  const char  *args[5] = {"x", "--nomp-backend", backend, "--nomp-device",
                          "bad"};
  const char **argv    = args;
  int          argc    = 5;

  // NOMP_DEVICE overrides the command line value, so it must be unset for the
  // invalid command line value to take effect.
  char *saved = test_setenv("NOMP_DEVICE", NULL);
  int   err   = nomp_init(&argc, &argv);
  test_restore_env("NOMP_DEVICE", saved);

  nomp_test_assert(nomp_get_err_no(err) == NOMP_USER_INPUT_IS_INVALID);
  nomp_test_assert(argc == 1);
  nomp_test_assert(strcmp(argv[0], "x") == 0);
  nomp_test_assert(argv[1] == NULL);

  return 0;
}

static int test_null_arguments(const char *backend) {
  const char  *args[2]   = {"foo", "bar"};
  const char **argv      = args;
  const char **null_argv = NULL;
  int          argc      = 2;

  char *saved = test_setenv("NOMP_BACKEND", backend);
  int   err   = nomp_init(NULL, &argv);
  if (!err) err = nomp_finalize();
  if (!err) err = nomp_init(&argc, NULL);
  if (!err) err = nomp_finalize();
  if (!err) err = nomp_init(&argc, &null_argv);
  if (!err) err = nomp_finalize();
  test_restore_env("NOMP_BACKEND", saved);
  nomp_test_check(err);

  nomp_test_assert(argc == 2);
  nomp_test_assert(argv[0] == args[0] && argv[1] == args[1]);

  return 0;
}

int main(int argc, const char *argv[]) {
  char backend[NOMP_TEST_MAX_BUFFER_SIZE + 1];
  for (unsigned i = 0; i < (unsigned)argc; i++) {
    if (strncmp(argv[i], "--nomp-backend", NOMP_TEST_MAX_BUFFER_SIZE) == 0) {
      assert(i + 1 < (unsigned)argc);
      strncpy(backend, argv[i + 1], NOMP_TEST_MAX_BUFFER_SIZE);
      break;
    }
  }

  int err = 0;
  err |= SUBTEST(test_valid_arguments, backend);
  err |= SUBTEST(test_ignore_non_argument_string, backend);
  err |= SUBTEST(test_invalid_argument_flag, backend);
  err |= SUBTEST(test_missing_argument, backend);
  err |= SUBTEST(test_unknown_option_removed, backend);
  err |= SUBTEST(test_no_removal_exact_size, backend);
  err |= SUBTEST(test_index_zero_option, backend);
  err |= SUBTEST(test_argc_one_not_parsed, backend);
  err |= SUBTEST(test_option_looking_value, backend);
  err |= SUBTEST(test_conditional_terminator_and_stale_tail, backend);
  err |= SUBTEST(test_no_removal_no_terminator, backend);
  err |= SUBTEST(test_missing_value_unchanged, backend);
  err |= SUBTEST(test_later_failure_leaves_compacted_argv, backend);
  err |= SUBTEST(test_null_arguments, backend);

  return err;
}
