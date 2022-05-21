#include "nomp.h"
#include <math.h>
#include <stdio.h>

static const char *_nomp_lpy_knl_src =
    "#define lid(N) ((int) get_local_id(N))\n"
    "#define gid(N) ((int) get_group_id(N))\n\n"
    "__kernel void __attribute__ ((reqd_work_group_size(1, 1, 1))) "
    "loopy_kernel(int const N, __global int *__restrict__ a, "
    "__global int const *__restrict__ b)\n"
    "{\n"
    "  for (int i = 0; i <= -1 + N; ++i)\n"
    "    a[i] = b[0];\n"
    "}";

static int vec_init(int N, int *a, int *b) {
  int err = nomp_map(a, 0, N, sizeof(int), NOMP_ALLOC);
  nomp_chk(err);
  err = nomp_map(b, 0, 1, sizeof(int), NOMP_H2D);
  nomp_chk(err);

  static int _nomp_lpy_knl_hndl = -1;
  err = nomp_jit(&_nomp_lpy_knl_hndl, _nomp_lpy_knl_src, "loopy_kernel");
  nomp_chk(err);

  size_t _nomp_lpy_knl_gsize[1] = {1};
  size_t _nomp_lpy_knl_lsize[1] = {1};
  err =
      nomp_run(_nomp_lpy_knl_hndl, 1, _nomp_lpy_knl_gsize, _nomp_lpy_knl_lsize,
               3, NOMP_SCALAR, &N, sizeof(N), NOMP_PTR, a, NOMP_PTR, b);
  nomp_chk(err);

  err = nomp_map(a, 0, N, sizeof(int), NOMP_D2H);
  nomp_chk(err);
  err = nomp_map(a, 0, N, sizeof(int), NOMP_FREE);
  nomp_chk(err);
  err = nomp_map(b, 0, 1, sizeof(int), NOMP_FREE);
  nomp_chk(err);

  return 0;
}

int main(int argc, char *argv[]) {
  char *backend = argc > 1 ? argv[1] : "opencl";
  int device_id = argc > 2 ? atoi(argv[2]) : 0;
  int platform_id = argc > 3 ? atoi(argv[3]) : 0;

  int err = nomp_init(backend, device_id, platform_id);
  nomp_chk(err);

  int a[10] = {0};
  int b[5] = {5, 5, 5, 5, 5};
  vec_init(7, a, b);

  int i;
  for (i = 0; i < 7; i++) {
    if (fabs(a[i] - b[0]) > 1e-10) {
      printf("err: (a[%d] = %d) != %d\n", i, a[i], b[0]);
      break;
    }
  }

  err = nomp_finalize();
  nomp_chk(err);

  return (i < 7);
}
