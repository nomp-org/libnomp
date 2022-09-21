#include "nomp.h"
#include <stdlib.h>

int main(int argc, char *argv[]) {
  char *backend = argc > 1 ? argv[1] : "opencl";
  int device_id = argc > 2 ? atoi(argv[2]) : 0;
  int platform_id = argc > 3 ? atoi(argv[3]) : 0;

  int err = nomp_init(backend, device_id, platform_id);
  nomp_chk(err);

  double a[10] = {0};
  int N = 10;
  const char *knl = "void foo(double *a, int N) {\n"
                    "  for (int i = 0; i < N; i++)\n"
                    "    a[i] = i;\n"
                    "}";

  // Calling nomp_jit with invalid functions should return an error.
  int id = -1;
  err = nomp_jit(&id, knl, NULL, "invalid-file:invalid_func", 2, "a,N",
                 NOMP_PTR, sizeof(double), a, NOMP_INTEGER, sizeof(int), &N);
  nomp_assert(err == NOMP_USER_CALLBACK_NOT_FOUND);

  err = nomp_jit(&id, knl, NULL, "nomp-api-200:transform", 2, "a,N", NOMP_PTR,
                 sizeof(double), a, NOMP_INTEGER, sizeof(int), &N);
  nomp_chk(err);

  err = nomp_finalize();
  nomp_chk(err);

  return 0;
}
