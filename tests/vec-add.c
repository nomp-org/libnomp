#include <math.h>
#include <nomp.h>
#include <stdio.h>

#define check_err_(err, file, line)                                            \
  do {                                                                         \
    if (err != 0) {                                                            \
      char buf[BUFSIZ];                                                        \
      nomp_err_str(err, buf, BUFSIZ);                                          \
      printf("%s:%d %s\n", file, line, buf);                                   \
      return 1;                                                                \
    }                                                                          \
  } while (0)

#define check_err(err) check_err_(err, __FILE__, __LINE__)

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

const char *vec_init_src =
    "#define lid(N) ((int) get_local_id(N))\n"
    "#define gid(N) ((int) get_group_id(N))\n\n"
    "__kernel void __attribute__ ((reqd_work_group_size(1, 1, 1))) "
    "vec_init(__global float *__restrict__ a)\n"
    "{\n"
    "  for (int i = 0; i <= 9; ++i)\n"
    "    a[i] = 42;\n"
    "}";

int vec_add(float *x, float *y, float *z, int handle) {
  int err = nomp_map(x, 0, 10, sizeof(float), NOMP_H2D, handle);
  check_err(err);
  err = nomp_map(y, 0, 10, sizeof(float), NOMP_H2D, handle);
  check_err(err);
  err = nomp_map(z, 0, 10, sizeof(float), NOMP_ALLOC, handle);
  check_err(err);

  size_t gsize[1] = {1};
  size_t lsize[1] = {1};
  static int vec_add_hndl = -1;
  err = nomp_run(&vec_add_hndl, vec_add_src, "vec_add", handle, 1, gsize, lsize,
                 3, NOMP_PTR, x, NOMP_PTR, y, NOMP_PTR, z);
  check_err(err);

  err = nomp_map(z, 0, 10, sizeof(float), NOMP_D2H, handle);
  check_err(err);

  return err;
}

int vec_init(float *a, int handle) {
  int err = nomp_map(a, 0, 10, sizeof(float), NOMP_ALLOC, handle);
  check_err(err);

  size_t lpy_knl_gsize[1] = {1};
  size_t lpy_knl_lsize[1] = {1};
  static int vec_init_hndl = -1;
  err = nomp_run(&vec_init_hndl, vec_init_src, "vec_init", handle, 1,
                 lpy_knl_gsize, lpy_knl_lsize, 1, NOMP_PTR, a);
  check_err(err);

  err = nomp_map(a, 0, 10, sizeof(float), NOMP_D2H, handle);
  check_err(err);

  return err;
}

int main() {
  int handle;
  int err = nomp_init(&handle, "opencl", 0, 0);
  check_err(err);

  float a[10];
  vec_init(a, handle);

  float *x = a;
  int i;
  for (i = 0; i < 10; ++i)
    x[i] = 1729;

  float y[10] = {3, 3, 3, 3, 3, 3, 3, 3, 3, 3};
  float z[10];
  vec_add(x, y, z, handle);

  err = 0;
  for (int i = 0; i < 10; ++i)
    if (fabs(z[i] - 1739) > 1e-10) {
      printf("z[%d] = %.2f\n", i, z[i]);
      err = 1;
      break;
    }

  return 0;
}

#undef check_err_
#undef check_err
