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

  err = nomp_update(a, 0, 20, sizeof(int), NOMP_TO);
  nomp_chk(err);
  err = nomp_update(b, 0, 20, sizeof(int), NOMP_TO);
  nomp_chk(err);

  const char *knl = "void foo(int *a, int *b, int N) {                      \n"
                    "  for (int i = 0; i < N; i++)                          \n"
                    "    a[i] = b[i];                                       \n"
                    "}                                                      \n";

  static int id = -1;
  const char *annotations[1] = {0},
             *clauses[3] = {"transform", "nomp-api-200:transform", 0};
  err = nomp_jit(&id, knl, annotations, clauses, 3, "a,b,N", NOMP_PTR,
                 sizeof(int), a, NOMP_PTR, sizeof(int), b, NOMP_INTEGER,
                 sizeof(int), &N);
  nomp_chk(err);

  err = nomp_run(id, NOMP_PTR, a, NOMP_PTR, b, NOMP_INTEGER, &N, sizeof(int));
  nomp_chk(err);

  err = nomp_update(a, 0, 20, sizeof(int), NOMP_FROM);
  nomp_chk(err);

  for (int i = 0; i < N; i++)
    nomp_assert(a[i] == b[i]);

  err = nomp_update(a, 0, 20, sizeof(int), NOMP_FREE);
  nomp_chk(err);
  err = nomp_update(b, 0, 20, sizeof(int), NOMP_FREE);
  nomp_chk(err);

  err = nomp_finalize();
  nomp_chk(err);

  return 0;
}
