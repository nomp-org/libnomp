#include <assert.h>
#include <math.h>
#include <nomp.h>
#include <stdio.h>

#define print_err(str)                                                         \
  do {                                                                         \
    if (err != 0) {                                                            \
      nomp_err_str(err, buf, BUFSIZ);                                          \
      printf(str, buf);                                                        \
      return 1;                                                                \
    }                                                                          \
  } while (0)

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

int main(int argc, char *argv[]) {
  double a[10] = {0};
  char buf[BUFSIZ];

  int handle;
  int err = nomp_init(&handle, "opencl", 0, 0);
  print_err("nomp_init failed: %s\n");

  err = nomp_map(a, 0, 10, sizeof(double), NOMP_ALLOC, handle);
  print_err("nomp_alloc failed: %s\n");

  const size_t global[3] = {10, 1, 1};
  const size_t local[3] = {1, 1, 1};
  int kernel = -1;
  err = nomp_run(&kernel, knl_str, "vec_init", handle, 3, global, local, 1,
                 NOMP_PTR, a);
  print_err("nomp_run failed: %s\n");

  err = nomp_map(a, 0, 10, sizeof(double), NOMP_D2H, handle);
  print_err("nomp_map failed: %s\n");

  int i, ret_val = 0;
  for (i = 0; i < 10; i++)
    if (fabs(a[i] - 42.0) > 1e-10) {
      printf("err: (a[%d] = %lf) != 42.0\n", i, a[i]);
      ret_val = 1;
      break;
    }

  err = nomp_finalize(&handle);
  print_err("nomp_finalize failed: %s\n");

  return ret_val;
}

#undef print_err
