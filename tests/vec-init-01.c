#include <math.h>
#include <nomp.h>
#include <stdio.h>

const char *_nomp_lpy_knl_src =
    "#define lid(N) ((int) get_local_id(N))\n"
    "#define gid(N) ((int) get_group_id(N))\n\n"
    "__kernel void __attribute__ ((reqd_work_group_size(1, 1, 1))) "
    "loopy_kernel(int const N, __global int *__restrict__ a, "
    "__global int const *__restrict__ b)\n"
    "{\n"
    "  for (int i = 0; i <= -1 + N; ++i)\n"
    "    a[i] = b[0];\n"
    "}";

const int foo(int N, int *a, int *b) {
  int err = nomp_map(a, 0, N, sizeof(int), NOMP_ALLOC);
  nomp_check_err(err);
  err = nomp_map(b, 0, 1, sizeof(int), NOMP_H2D);
  nomp_check_err(err);

  size_t _nomp_lpy_knl_gsize[1] = {1};
  size_t _nomp_lpy_knl_lsize[1] = {1};
  static int _nomp_lpy_knl_hndl = -1;
  err = nomp_run(&_nomp_lpy_knl_hndl, _nomp_lpy_knl_src, "loopy_kernel", 1,
                 _nomp_lpy_knl_gsize, _nomp_lpy_knl_lsize, 3, NOMP_INT, N,
                 NOMP_PTR, a, NOMP_PTR, b);
  nomp_check_err(err);

  err = nomp_map(a, 0, N, sizeof(int), NOMP_D2H);
  nomp_check_err(err);
  err = nomp_map(a, 0, N, sizeof(int), NOMP_FREE);
  nomp_check_err(err);
  err = nomp_map(b, 0, 1, sizeof(int), NOMP_FREE);
  nomp_check_err(err);

  return 0;
}

int main(int argc, char *argv[]) {
  int err = nomp_init("opencl", 0, 0);
  nomp_check_err(err);

  int a[10] = {0};
  int b[5] = {5, 5, 5, 5, 5};

  foo(7, a, b);

  err = 0;
  int i;
  for (i = 0; i < 7; i++) {
    if (fabs(a[i] - b[0]) > 1e-10) {
      printf("err: (a[%d] = %d) != %d\n", i, a[i], b[0]);
      err = 1;
      break;
    }
  }

  err = nomp_finalize();
  nomp_check_err(err);

  return err;
}
