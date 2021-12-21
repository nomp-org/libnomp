#include <math.h>
#include <nomp.h>
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

const int foo(int N, double *a) {
  int err = nomp_map(a, 0, 10, sizeof(double), NOMP_ALLOC);
  nomp_check_err(err);

  size_t _nomp_lpy_knl_gsize[1] = {1};
  size_t _nomp_lpy_knl_lsize[1] = {1};
  static int _nomp_lpy_knl_hndl = -1;
  err = nomp_run(&_nomp_lpy_knl_hndl, _nomp_lpy_knl_src, "loopy_kernel", 1,
                 _nomp_lpy_knl_gsize, _nomp_lpy_knl_lsize, 1, NOMP_PTR, a);
  nomp_check_err(err);

  err = nomp_map(a, 0, 10, sizeof(double), NOMP_D2H);
  nomp_check_err(err);

  err = nomp_map(a, 0, 10, sizeof(double), NOMP_FREE);
  nomp_check_err(err);

  return 0;
}

int main(int argc, char *argv[]) {
  int err = nomp_init("opencl", 0, 0);
  nomp_check_err(err);

  double a[10] = {0};

  foo(10, a);

  err = 0;
  int i;
  for (i = 0; i < 10; i++) {
    if (fabs(a[i] - 42.0) > 1e-10) {
      printf("err: (a[%d] = %lf) != 42.0\n", i, a[i]);
      err = 1;
      break;
    }
  }

  err = nomp_finalize();
  nomp_check_err(err);

  return err;
}
