#include "nomp.h"
#include <math.h>
#include <stdlib.h>

void foo(double *a, int n) {
  int err = nomp_update(a, 0, n, sizeof(double), NOMP_TO);
  nomp_chk(err);

  static int id = -1;
  const char knl[96] = "void loopy_kernel(double *a, int n) {\nfor "
                       "(unsigned i = 0; i < n; i++)\n    a[i] = i;\n}";
  const char args[4] = "a,n";
  const char *annotations[1] = {0},
             *clauses[3] = {"transform", "nomp-api-200:transform", 0};
  err = nomp_jit(&id, knl, annotations, clauses, 2, args, NOMP_PTR,
                 sizeof(double), a, NOMP_INTEGER, sizeof(int), &n);
  nomp_chk(err);

  err = nomp_run(id, NOMP_PTR, a, NOMP_INTEGER, &n, sizeof(int));
  nomp_chk(err);

  err = nomp_update(a, 0, n, sizeof(double), NOMP_FROM);
  nomp_chk(err);
}

int main(int argc, char *argv[]) {
  char *backend = argc > 1 ? argv[1] : "opencl";
  int device_id = argc > 2 ? atoi(argv[2]) : 0;
  int platform_id = argc > 3 ? atoi(argv[3]) : 0;

  int err = nomp_init(backend, device_id, platform_id);
  nomp_chk(err);

  double a[10] = {0};
  foo(a, 10);

  for (int i = 0; i < 10; i++)
    nomp_assert(fabs(a[i] - i) < 1e-12);

  err = nomp_finalize();
  nomp_chk(err);

  return 0;
}
