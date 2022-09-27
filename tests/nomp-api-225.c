#include "nomp.h"
#include <math.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
  char *backend = argc > 1 ? argv[1] : "opencl";
  int device_id = argc > 2 ? atoi(argv[2]) : 0;
  int platform_id = argc > 3 ? atoi(argv[3]) : 0;

  int err = nomp_init(backend, device_id, platform_id);
  nomp_chk(err);

  float a[20] = {0}, b[20] = {1, 2, 3, 4, 5};
  int N = 20;

  err = nomp_map(a, 0, 20, sizeof(float), NOMP_H2D);
  nomp_chk(err);
  err = nomp_map(b, 0, 20, sizeof(float), NOMP_H2D);
  nomp_chk(err);

  const char *knl = "void foo(float *a, float *b, int N) {\n"
                    "  for (int i = 0; i < N; i++)\n"
                    "    a[i] = 2 * b[i] + 1;\n"
                    "}";

  int id = -1;
  err = nomp_jit(&id, knl, NULL, "nomp-api-200:transform", 3, "a,b,N", NOMP_PTR,
                 sizeof(float), a, NOMP_PTR, sizeof(float), b, NOMP_INTEGER,
                 sizeof(int), &N);
  nomp_chk(err);

  err = nomp_run(id, NOMP_PTR, a, NOMP_PTR, b, NOMP_INTEGER, &N, sizeof(int));
  nomp_chk(err);

  err = nomp_map(a, 0, 20, sizeof(float), NOMP_D2H);
  nomp_chk(err);

  for (int i = 0; i < N; i++)
    nomp_assert(fabs(a[i] - 2 * b[i] - 1) < 1e-08);

  err = nomp_map(a, 0, 20, sizeof(float), NOMP_FREE);
  nomp_chk(err);
  err = nomp_map(b, 0, 20, sizeof(float), NOMP_FREE);
  nomp_chk(err);

  err = nomp_finalize();
  nomp_chk(err);

  return 0;
}
