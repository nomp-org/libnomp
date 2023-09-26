#include "nomp-test.h"

// Invoke with invalid kernel_id
static int test_invalid_kernel_id(int argc, const char **argv, int *id, int *a,
                                  int *b, int n) {
  const char *knl = "void foo(int *a, int *b, int N) {                      \n"
                    "  for (int i = 0; i < N; i++)                          \n"
                    "    a[i] = a[i] * b[i];                                \n"
                    "}                                                      \n";

  nomp_test_check(nomp_init(argc, argv));
  nomp_test_check(nomp_update(a, 0, n, sizeof(int), NOMP_TO));

  const char *clauses[4] = {"transform", "nomp_api_100", "tile", 0};
  nomp_test_check(nomp_jit(id, knl, clauses, 3, "a", sizeof(int *), NOMP_PTR,
                           "b", sizeof(int *), NOMP_PTR, "N", sizeof(int),
                           NOMP_INT));

  int err = nomp_run(-1, a, b, &n);
  nomp_test_assert(nomp_get_err_no(err) == NOMP_USER_INPUT_IS_INVALID);
  char *log = nomp_get_err_str(err);
  int eq = logcmp(log, "\\[Error\\] .*\\/src\\/nomp.[c|cpp]:[0-9]* Kernel "
                       "id -1 passed to nomp_run is not valid.");
  nomp_free(&log);
  nomp_test_assert(eq);

  return 0;
}

// Invoke fails because b is not mapped
static int test_unmapped_variable(int id, int *a, int *b, int n) {
  int err = nomp_run(id, a, b, &n);
  nomp_test_assert(nomp_get_err_no(err) == NOMP_USER_MAP_PTR_IS_INVALID);

  char *desc = nomp_get_err_str(err);
  int eq = logcmp(desc, "\\[Error\\] .*\\/src\\/.*.[c|cpp]:[0-9]* Map pointer "
                        "0[xX][0-9a-fA-F]* was not found on device.");
  nomp_free(&desc);
  nomp_test_assert(eq);

  nomp_test_check(nomp_finalize());

  return 0;
}

int main(int argc, const char *argv[]) {
  const int n = 10;
  int a[10], b[10];
  for (int i = 0; i < n; i++)
    a[i] = n - i, b[i] = i;

  int err = 0, id = -1;

  err |= SUBTEST(test_invalid_kernel_id, argc, argv, &id, a, b, n);
  err |= SUBTEST(test_unmapped_variable, id, a, b, n);

  return err;
}
