#include "nomp.h"
#include <math.h>
#include <stdlib.h>

void foo(float *a, float *b, int n) {
  static int id = -1;
  const char *knl = "void loopy_kernel(float *a, float *b, int n) {         \n"
                    "for (unsigned i = 0; i < n; i++)                       \n"
                    "  a[i] = b[i] + a[i];                                  \n"
                    "}                                                      \n";
  const char *annotations[1] = {0},
             *clauses[3] = {"transform", "nomp-api-200:transform", 0};
  int err = nomp_jit(&id, knl, annotations, clauses, 3, "a,b,n", NOMP_PTR,
                     sizeof(float), a, NOMP_PTR, sizeof(float), b, NOMP_INTEGER,
                     sizeof(int), &n);
  nomp_chk(err);

  err = nomp_run(id, NOMP_PTR, a, NOMP_PTR, b, NOMP_INTEGER, &n, sizeof(int));
  nomp_chk(err);
}

int main(int argc, char *argv[]) {
  char *backend = argc > 1 ? argv[1] : "opencl";
  int device_id = argc > 2 ? atoi(argv[2]) : 0;
  int platform_id = argc > 3 ? atoi(argv[3]) : 0;

  int err = nomp_init(backend, device_id, platform_id);
  nomp_chk(err);

  float a[10], b[10];
  int n = 10;

  for (unsigned i = 0; i < 10; i++) {
    a[i] = n - i;
    b[i] = i;
  }

  err = nomp_update(a, 0, n, sizeof(float), NOMP_TO);
  nomp_chk(err);
  err = nomp_update(b, 0, n, sizeof(float), NOMP_TO);
  nomp_chk(err);

  foo(a, b, n);

  err = nomp_update(a, 0, n, sizeof(float), NOMP_FROM);
  nomp_chk(err);

  for (int i = 0; i < 10; i++)
    nomp_assert(fabs(a[i] - n) < 1e-8);

  err = nomp_finalize();
  nomp_chk(err);

  return 0;
}
