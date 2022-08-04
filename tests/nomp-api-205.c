#include "nomp.h"
#include <stdlib.h>

int main(int argc, char *argv[]) {
  char *backend = argc > 1 ? argv[1] : "opencl";
  int device_id = argc > 2 ? atoi(argv[2]) : 0;
  int platform_id = argc > 3 ? atoi(argv[3]) : 0;

  int err = nomp_init(backend, device_id, platform_id);
  nomp_chk(err);

  double a[20] = {0}, b[20] = {1, 2, 3, 4, 5};
  int N = 20;
  const char *knl = "void foo(double *a, double *b, int N) {\n"
                    "  for (int i = 0; i < N; i++)\n"
                    "    a[i] = b[i];\n"
                    "}";

  int id = -1, ndim = -1;
  size_t global[3], local[3];
  err = nomp_jit(&id, &ndim, global, local, knl, NULL, "nomp-api-205:transform",
                 3, "a,b,N", NOMP_PTR, sizeof(double), a, NOMP_PTR,
                 sizeof(double), b, NOMP_INTEGER, sizeof(int), &N);
  nomp_chk(err);
  nomp_assert(global[0] == 20);

  err = nomp_finalize();
  nomp_chk(err);

  return 0;
}
