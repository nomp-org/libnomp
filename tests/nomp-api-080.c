#include "nomp-test.h"
#include "nomp.h"
#include <limits.h>

int main(int argc, const char *argv[]) {

  // Ignore invaild arguments
  const char *args1[] = {"foo.c", "-b", "opencl", "-d", "0", "-p", "0"};
  argc = 7;
  int err = nomp_init(argc, args1);
  nomp_chk(err);
  err = nomp_finalize();
  nomp_chk(err);

  const char *args2[] = {"-b", "opencl", "cuda", "-d", "0", "1", "-p", "0"};
  argc = 8;
  err = nomp_init(argc, args2);
  nomp_chk(err);
  err = nomp_finalize();
  nomp_chk(err);

  // Invaild argument should return an error
  const char *args3[] = {"--b", "opencl", "-d", "0", "-p", "0"};
  argc = 6;
  err = nomp_init(argc, args3);
  nomp_assert(nomp_get_log_no(err) == NOMP_USER_ARGS_IS_INVALID);
  err = nomp_finalize();
  nomp_assert(nomp_get_log_no(err) == NOMP_RUNTIME_NOT_INITIALIZED);

  // Missing value should return an error
  const char *args4[] = {"-b", "opencl", "-d", "0", "-p"};
  argc = 5;
  err = nomp_init(argc, args4);
  nomp_assert(nomp_get_log_no(err) == NOMP_USER_ARGS_IS_INVALID);
  err = nomp_finalize();
  nomp_assert(nomp_get_log_no(err) == NOMP_RUNTIME_NOT_INITIALIZED);

  return 0;
}
