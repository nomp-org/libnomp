#include <math.h>
#include <nomp.h>
#include <stdio.h>

const char *_nomp_lpy_knl_src =
    "#define lid(N) ((int) get_local_id(N))\n"
    "#define gid(N) ((int) get_group_id(N))\n\n"
    "__kernel void __attribute__ ((reqd_work_group_size(1, 1, 1))) "
    "loopy_kernel(int const N, __global float *__restrict__ a, "
    "__global float const *__restrict__ b)\n"
    "{\n"
    "  for (int i = 0; i <= -1 + N; ++i)\n"
    "    a[i] = b[i];\n"
    "}";

const int vec_init(int N, float *a, float *b) {
  int err = nomp_map(a, 0, N, sizeof(float), NOMP_ALLOC);
  nomp_check_err(err);
  err = nomp_map(b, 0, N, sizeof(float), NOMP_H2D);
  nomp_check_err(err);

  size_t _nomp_lpy_knl_gsize[1] = {1};
  size_t _nomp_lpy_knl_lsize[1] = {1};
  static int _nomp_lpy_knl_hndl = -1;
  err = nomp_run(&_nomp_lpy_knl_hndl, _nomp_lpy_knl_src, "loopy_kernel", 1,
                 _nomp_lpy_knl_gsize, _nomp_lpy_knl_lsize, 3, NOMP_SCALAR, &N,
                 sizeof(N), NOMP_PTR, a, NOMP_PTR, b);
  nomp_check_err(err);

  err = nomp_map(a, 0, N, sizeof(int), NOMP_D2H);
  nomp_check_err(err);
  err = nomp_map(a, 0, N, sizeof(double), NOMP_FREE);
  nomp_check_err(err);
  err = nomp_map(b, 0, N, sizeof(double), NOMP_FREE);
  nomp_check_err(err);

  return 0;
}

int main(int argc, char *argv[]) {
  int err = nomp_init("opencl", 0, 0);
  nomp_check_err(err);

  float a[10] = {0.0};
  float b[6] = {5.0, 4.0, 3.0, 2.0, 1.0, 0.0};

  vec_init(6, a, b);

  int i;
  for (i = err = 0; err == 0 && i < 6; i++)
    if (err = (fabs(a[i] - b[i]) > 1e-10))
      printf("err: (a[%d] = %lf) != %lf (= b[%d])\n", i, a[i], b[i], i);

  err = nomp_finalize();
  nomp_check_err(err);

  return err;
}
