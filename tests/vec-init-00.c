#include <math.h>
#include <nomp.h>
#include <stdio.h>

const char *knl_str =
    "#define lid(N) ((int) get_local_id(N))\n"
    "#define gid(N) ((int) get_group_id(N))\n"
    "#if __OPENCL_C_VERSION__ < 120\n"
    "#pragma OPENCL EXTENSION cl_khr_fp64: enable\n"
    "#endif\n\n"
    "__kernel void __attribute__ ((reqd_work_group_size(1, 1, 1))) "
    "vec_init(__global double *__restrict__ a)\n"
    "{\n"
    "for (int i = 0; i <= 9; ++i)\n"
    "a[i] = 42.0;\n"
    "}";

const int foo(int N, double *a) {
  int err = nomp_map(a, 0, 10, sizeof(double), NOMP_ALLOC);
  nomp_check_err(err);

  const size_t global[3] = {10, 1, 1};
  const size_t local[3] = {1, 1, 1};
  static int kernel = -1;
  err =
      nomp_run(&kernel, knl_str, "vec_init", 3, global, local, 1, NOMP_PTR, a);
  nomp_check_err(err);

  err = nomp_map(a, 0, 10, sizeof(double), NOMP_D2H);
  nomp_check_err(err);

  err = nomp_map(a, 0, 10, sizeof(double), NOMP_FREE);
  nomp_check_err(err);
}

int main(int argc, char *argv[]) {
  int err = nomp_init("opencl", 0, 0);
  nomp_check_err(err);

  double a[10] = {0};

  foo(10, a);

  err = 0;
  int i;
  for (i = 0; i < 10; i++)
    if (fabs(a[i] - 42.0) > 1e-10) {
      printf("err: (a[%d] = %lf) != 42.0\n", i, a[i]);
      err = 1;
      break;
    }

  err = nomp_finalize();
  nomp_check_err(err);

  return err;
}
