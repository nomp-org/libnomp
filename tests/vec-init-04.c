#include "nomp.h"
#include <math.h>
#include <stdio.h>

const char *_nomp_lpy_knl_src =
    "#define lid(N) ((int) get_local_id(N))\n"
    "#define gid(N) ((int) get_group_id(N))\n\n"
    "__kernel void __attribute__ ((reqd_work_group_size(1, 1, 1))) "
    "loopy_kernel(int const N, int const offa, int const offb, "
    "__global double *__restrict__ a, "
    "__global double const *__restrict__ b)\n"
    "{\n"
    "  for (int i = 0; i <= -1 + N; ++i)\n"
    "    a[offa + i] = b[offb + i];\n"
    "}";

const int vec_init(int N, int offa, int offb, double *a, double *b) {
  int err = nomp_map(a, 0, N + offa, sizeof(double), NOMP_ALLOC);
  nomp_chk(err);
  err = nomp_map(b, 0, N + offb, sizeof(double), NOMP_H2D);
  nomp_chk(err);

  static int _nomp_lpy_knl_hndl = -1;
  err = nomp_jit(&_nomp_lpy_knl_hndl, _nomp_lpy_knl_src, "loopy_kernel");
  nomp_chk(err);

  size_t _nomp_lpy_knl_gsize[1] = {1};
  size_t _nomp_lpy_knl_lsize[1] = {1};
  err =
      nomp_run(_nomp_lpy_knl_hndl, 1, _nomp_lpy_knl_gsize, _nomp_lpy_knl_lsize,
               5, NOMP_SCALAR, &N, sizeof(N), NOMP_SCALAR, &offa, sizeof(offa),
               NOMP_SCALAR, &offb, sizeof(offb), NOMP_PTR, a, NOMP_PTR, b);
  nomp_chk(err);

  err = nomp_map(a, 0, N + offa, sizeof(double), NOMP_D2H);
  nomp_chk(err);
  err = nomp_map(a, 0, N + offa, sizeof(double), NOMP_FREE);
  nomp_chk(err);
  err = nomp_map(b, 0, N + offb, sizeof(double), NOMP_FREE);
  nomp_chk(err);

  return 0;
}

int main(int argc, char *argv[]) {
  char *backend = argc > 1 ? argv[1] : "opencl";
  int device_id = argc > 2 ? atoi(argv[2]) : 0;
  int platform_id = argc > 3 ? atoi(argv[3]) : 0;

  int err = nomp_init(backend, device_id, platform_id);
  nomp_chk(err);

  double a[9] = {0.0};
  double b[6] = {5.0, 4.0, 3.0, 2.0, 1.0, 0.0};
  vec_init(3, 1, 2, a, b);

  int i;
  for (i = 0; i < 3; i++) {
    if (fabs(a[1 + i] - b[2 + i]) > 1e-10) {
      printf("err: (a[%d] = %lf) != %lf (= b[%d])\n", 3 + i, a[1 + i], b[2 + i],
             i);
      break;
    }
  }

  err = nomp_finalize();
  nomp_chk(err);

  return (i < 3);
}
