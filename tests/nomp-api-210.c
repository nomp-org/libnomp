#include "nomp.h"
#include <stdlib.h>

int main(int argc, char *argv[]) {
  char *backend = argc > 1 ? argv[1] : "opencl";
  int device_id = argc > 2 ? atoi(argv[2]) : 0;
  int platform_id = argc > 3 ? atoi(argv[3]) : 0;

  int err = nomp_init(backend, device_id, platform_id);
  nomp_chk(err);

  int a[20] = {0}, b[20] = {1, 2, 3, 4, 5};
  int N = 20;

  err = nomp_map(a, 0, 20, sizeof(int), NOMP_H2D);
  nomp_chk(err);
  err = nomp_map(b, 0, 20, sizeof(int), NOMP_H2D);
  nomp_chk(err);

  const char *knl = "void foo(int *a, int *b, int N) {\n"
                    "  for (int i = 0; i < N; i++)\n"
                    "    a[i] = b[i];\n"
                    "}";

  int id = -1;
  err = nomp_jit(&id, knl, NULL, "nomp-api-200:transform", 3, "a,b,N", NOMP_PTR,
                 sizeof(int), a, NOMP_PTR, sizeof(int), b, NOMP_INTEGER,
                 sizeof(int), &N);
  nomp_chk(err);

  // FIXME: Fix the order of argments to the loopy kernel to match
  // nomp_jit()
  err =
      nomp_run(id, 3, NOMP_INTEGER, &N, sizeof(int), NOMP_PTR, a, NOMP_PTR, b);
  nomp_chk(err);

  err = nomp_map(a, 0, 20, sizeof(int), NOMP_D2H);
  nomp_chk(err);

  for (int i = 0; i < N; i++)
    nomp_assert(a[i] == b[i]);

  err = nomp_map(a, 0, 20, sizeof(int), NOMP_FREE);
  nomp_chk(err);
  err = nomp_map(b, 0, 20, sizeof(int), NOMP_FREE);
  nomp_chk(err);

  err = nomp_finalize();
  nomp_chk(err);

  return 0;
}
