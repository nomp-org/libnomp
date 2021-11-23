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
  int err = nomp_init("opencl", 0, 0);
  check_err(err);

  double a[10] = {0};
  err = nomp_map(a, 0, 10, sizeof(double), NOMP_ALLOC);
  check_err(err);

  const size_t global[3] = {10, 1, 1};
  const size_t local[3] = {1, 1, 1};
  int kernel = -1;
  err =
      nomp_run(&kernel, knl_str, "vec_init", 3, global, local, 1, NOMP_PTR, a);
  check_err(err);

  err = nomp_map(a, 0, 10, sizeof(double), NOMP_D2H);
  check_err(err);

  err = 0;
  int i;
  for (i = 0; i < 10; i++)
    if (fabs(a[i] - 42.0) > 1e-10) {
      printf("err: (a[%d] = %lf) != 42.0\n", i, a[i]);
      err = 1;
      break;
    }

  err = nomp_map(a, 0, 10, sizeof(double), NOMP_FREE);
  check_err(err);

  err = nomp_finalize();
  check_err(err);

  return err;
}

#undef check_err_
#undef check_err
