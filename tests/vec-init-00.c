#include "nomp.h"
#include <math.h>
#include <stdio.h>

const char *_nomp_lpy_knl_src =
    "#define lid(N) ((int) get_local_id(N))\n"
    "#define gid(N) ((int) get_group_id(N))\n"
    "__kernel void __attribute__ ((reqd_work_group_size(1, 1, 1))) "
    "loopy_kernel(__global double *__restrict__ a)\n"
    "{\n"
    "for (int i = 0; i <= 9; ++i)\n"
    "a[i] = 42.0;\n"
    "}";

const int vec_init(int N, double *a) {
  int err = nomp_map(a, 0, 10, sizeof(double), NOMP_ALLOC);
  nomp_chk(err);

  static int _nomp_lpy_knl_hndl = -1;
  err = nomp_jit(&_nomp_lpy_knl_hndl, _nomp_lpy_knl_src, "loopy_kernel");
  nomp_chk(err);

  size_t _nomp_lpy_knl_gsize[1] = {1};
  size_t _nomp_lpy_knl_lsize[1] = {1};
  err = nomp_run(_nomp_lpy_knl_hndl, 1, _nomp_lpy_knl_gsize,
                 _nomp_lpy_knl_lsize, 1, NOMP_PTR, a);
  nomp_chk(err);

  err = nomp_map(a, 0, 10, sizeof(double), NOMP_D2H);
  nomp_chk(err);
  err = nomp_map(a, 0, 10, sizeof(double), NOMP_FREE);
  nomp_chk(err);

  return 0;
}

int main(int argc, char *argv[]) {
  char *backend = argc > 1 ? argv[1] : "opencl";
  int device_id = argc > 2 ? atoi(argv[2]) : 0;
  int platform_id = argc > 3 ? atoi(argv[3]) : 0;

  int err = nomp_init(backend, device_id, platform_id);
  nomp_chk(err);

  double a[10] = {0};
  vec_init(10, a);

  int i;
  for (i = 0; i < 10; i++) {
    if (fabs(a[i] - 42.0) > 1e-10) {
      printf("err: (a[%d] = %lf) != 42.0\n", i, a[i]);
      break;
    }
  }

  err = nomp_finalize();
  nomp_chk(err);

  return (i < 10);
}
