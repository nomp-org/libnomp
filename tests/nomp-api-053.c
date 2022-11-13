#include "nomp-test.h"
#include "nomp.h"

int main(int argc, char *argv[]) {
  char *backend = argc > 1 ? argv[1] : "opencl";
  int device = argc > 2 ? atoi(argv[2]) : 0;
  int platform = argc > 3 ? atoi(argv[3]) : 0;
  int n = 10;
  int a[10], b[10];
  for (unsigned i = 0; i < n; i++)
    a[i] = n - i, b[i] = i;

  const char *knl = "void foo(int *a, int *b, int N) {                      \n"
                    "  for (int i = 0; i < N; i++)                          \n"
                    "    a[i] = a[i] * b[i];                                \n"
                    "}                                                      \n";

  int err = nomp_init(backend, platform, device);
  err = nomp_update(a, 0, n, sizeof(int), NOMP_TO);
  nomp_chk(err);

  static int id = -1;
  const char *clauses[4] = {"transform", "nomp-api-50", "transform", 0};
  err = nomp_jit(&id, knl, clauses);
  nomp_chk(err);

  // Invoke with invalid kernel_id
  err = nomp_run(-1, 3, "a", NOMP_PTR, sizeof(int), a, "b", NOMP_PTR,
                 sizeof(int), b, "N", NOMP_INTEGER, sizeof(int), &n);
  nomp_assert(nomp_get_log_no(err) == NOMP_USER_INPUT_NOT_VALID);
  char *desc;
  err = nomp_get_log_str(&desc, err);
  int matched = match_log(desc, "\\[Error\\] .*\\/src\\/nomp.c:[0-9]* Kernel "
                                "id -1 passed to nomp_run is not valid.");
  nomp_assert(matched);
  tfree(desc);

  // Invoke fails because b is not mapped
  err = nomp_run(id, 3, "a", NOMP_PTR, sizeof(int), a, "b", NOMP_PTR,
                 sizeof(int), b, "N", NOMP_INTEGER, sizeof(int), &n);
  nomp_assert(nomp_get_log_no(err) == NOMP_USER_MAP_PTR_NOT_VALID);
  err = nomp_get_log_str(&desc, err);
  matched = match_log(desc, "\\[Error\\] .*\\/src\\/.*.c:[0-9]* Map pointer "
                            "0[xX][0-9a-fA-F]* was not found on device.");
  nomp_assert(matched);
  tfree(desc);

  err = nomp_finalize();
  nomp_chk(err);

  return 0;
}
