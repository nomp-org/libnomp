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

  int id = -1, ndim = -1;
  size_t global[3], local[3];
  err = nomp_jit(&id, &ndim, global, local, knl, NULL, "nomp-api-30:transform",
                 2, "a,N", 1, sizeof(double), a, 0, sizeof(int), &N);
  nomp_chk(err);

  err = nomp_finalize();
  nomp_chk(err);

  return 0;
}
