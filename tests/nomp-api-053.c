#include "nomp-test.h"

const int n = 10;

// Invoke with invalid kernel_id
static int test_invalid_kernel_id(int argc, const char **argv, int *id, int *a,
                                  int *b) {
  const char *knl = "void foo(int *a, int *b, int N) {                      \n"
                    "  for (int i = 0; i < N; i++)                          \n"
                    "    a[i] = a[i] * b[i];                                \n"
                    "}                                                      \n";

  int err = nomp_init(argc, argv);
  nomp_test_chk(err);
  err = nomp_update(a, 0, n, sizeof(int), NOMP_TO);
  nomp_test_chk(err);

  const char *clauses[4] = {"transform", "nomp-api-50", "transform", 0};
  err = nomp_jit(id, knl, clauses);
  nomp_test_chk(err);

  err = nomp_run(-1, 3, "a", NOMP_PTR, sizeof(int), a, "b", NOMP_PTR,
                 sizeof(int), b, "N", NOMP_INTEGER, sizeof(int), &n);
  nomp_test_assert(nomp_get_log_no(err) == NOMP_USER_INPUT_IS_INVALID);
  char *desc;
  nomp_get_log_str(&desc, err);
  int matched = match_log(desc, "\\[Error\\] .*\\/src\\/nomp.c:[0-9]* Kernel "
                                "id -1 passed to nomp_run is not valid.");
  nomp_test_assert(matched);
  tfree(desc);
  return 0;
}

// Invoke fails because b is not mapped
static int test_unmapped_variable(int id, int *a, int *b) {
  char *desc;
  int err = nomp_run(id, 3, "a", NOMP_PTR, sizeof(int), a, "b", NOMP_PTR,
                     sizeof(int), b, "N", NOMP_INTEGER, sizeof(int), &n);
  nomp_test_assert(nomp_get_log_no(err) == NOMP_USER_MAP_PTR_IS_INVALID);
  nomp_get_log_str(&desc, err);
  int matched =
      match_log(desc, "\\[Error\\] .*\\/src\\/.*.c:[0-9]* Map pointer "
                      "0[xX][0-9a-fA-F]* was not found on device.");
  nomp_test_assert(matched);
  tfree(desc);

  err = nomp_finalize();
  nomp_test_chk(err);
  return 0;
}

int main(int argc, const char *argv[]) {
  static int a[10], b[10];
  static int id = -1;
  int err = 0;

  for (int i = 0; i < n; i++) {
    a[i] = n - i, b[i] = i;
  }

  err |= SUBTEST(test_invalid_kernel_id, argc, argv, &id, a, b);
  err |= SUBTEST(test_unmapped_variable, id, a, b);

  return err;
}
