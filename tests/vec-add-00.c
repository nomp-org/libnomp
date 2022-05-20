#include "nomp.h"
#include <math.h>
#include <stdio.h>

const char *vec_add_src =
    "#define lid(N) ((int) get_local_id(N))\n"
    "#define gid(N) ((int) get_group_id(N))\n\n"
    "__kernel void __attribute__ ((reqd_work_group_size(1, 1, 1))) "
    "vec_add(__global float const *__restrict__ x,"
    "__global float const *__restrict__ y, __global float *__restrict__ z)\n"
    "{\n"
    " for (int i = 0; i <= 9; ++i)\n"
    "    z[i] = y[i] + x[i];\n"
    "}";

int vec_add(float *x, float *y, float *z) {
  int err = nomp_map(x, 0, 10, sizeof(float), NOMP_H2D);
  nomp_chk(err);
  err = nomp_map(y, 0, 10, sizeof(float), NOMP_H2D);
  nomp_chk(err);
  err = nomp_map(z, 0, 10, sizeof(float), NOMP_ALLOC);
  nomp_chk(err);

  static int vec_add_hndl = -1;
  err = nomp_jit(&vec_add_hndl, vec_add_src, "vec_add");
  nomp_chk(err);

  size_t gsize[1] = {1};
  size_t lsize[1] = {1};
  err = nomp_run(vec_add_hndl, 1, gsize, lsize, 3, NOMP_PTR, x, NOMP_PTR, y,
                 NOMP_PTR, z);
  nomp_chk(err);

  err = nomp_map(z, 0, 10, sizeof(float), NOMP_D2H);
  nomp_chk(err);
  err = nomp_map(z, 0, 10, sizeof(float), NOMP_FREE);
  nomp_chk(err);
  err = nomp_map(y, 0, 10, sizeof(float), NOMP_FREE);
  nomp_chk(err);
  err = nomp_map(x, 0, 10, sizeof(float), NOMP_FREE);
  nomp_chk(err);

  return err;
}

int main(int argc, char *argv[]) {
  char *backend = argc > 1 ? argv[1] : "opencl";
  int device_id = argc > 2 ? atoi(argv[2]) : 0;
  int platform_id = argc > 3 ? atoi(argv[3]) : 0;

  int err = nomp_init(backend, device_id, platform_id);
  nomp_chk(err);

  float x[10] = {1729, 1729, 1729, 1729, 1729, 1729, 1729, 1729, 1729, 1729};
  float y[10] = {3, 3, 3, 3, 3, 3, 3, 3, 3, 3};
  float z[10];
  vec_add(x, y, z);

  int i;
  for (i = err = 0; err == 0 && i < 10; ++i)
    if (err = (fabs(z[i] - 1732) > 1e-10))
      printf("z[%d] = %f ! 1732\n", i, z[i]);

  return err;
}
