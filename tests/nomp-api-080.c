#include "nomp-test.h"
#include "nomp.h"
#include <limits.h>

int main(int argc, char *argv[]) {

  // Missing flag should return an error
  const char *args[] = {" ", "opencl", "-d", "0", "-p", "0"};
  argc = 6;
  int err = nomp_init(argc, args);
  nomp_assert(nomp_get_log_no(err) == NOMP_USER_ARGS_IS_INVALID);
  err = nomp_finalize();
  nomp_assert(nomp_get_log_no(err) == NOMP_RUNTIME_NOT_INITIALIZED);

  // Missing flag should return an error
  const char *args2[] = {" ", "-b", "opencl", "-d", "0", "1"};
  argc = 6;
  err = nomp_init(argc, args2);
  nomp_assert(nomp_get_log_no(err) == NOMP_USER_ARGS_IS_INVALID);
  err = nomp_finalize();
  nomp_assert(nomp_get_log_no(err) == NOMP_RUNTIME_NOT_INITIALIZED);

  // Missing value should return an error
  const char *args3[] = {" ", "-b", "opencl", "-d", "0", "-p"};
  argc = 6;
  err = nomp_init(argc, args3);
  nomp_assert(nomp_get_log_no(err) == NOMP_USER_PLATFORM_IS_INVALID);
  err = nomp_finalize();
  nomp_assert(nomp_get_log_no(err) == NOMP_RUNTIME_NOT_INITIALIZED);

  return 0;
}
