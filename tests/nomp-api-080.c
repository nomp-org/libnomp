#include "nomp-test.h"
#include "nomp.h"
#include <limits.h>

int main(int argc, const char *argv[]) {

  // Invaild argument should return an error
  const char *args[] = {"--b", "opencl", "-d", "0", "-p", "0"};
  argc = 6;
  int err = nomp_init(argc, args);
  nomp_assert(nomp_get_log_no(err) == NOMP_USER_ARGS_IS_INVALID);
  err = nomp_finalize();
  nomp_assert(nomp_get_log_no(err) == NOMP_RUNTIME_NOT_INITIALIZED);

  // Missing value should return an error
  const char *args3[] = {"-b", "opencl", "-d", "0", "-p"};
  argc = 5;
  err = nomp_init(argc, args3);
  nomp_assert(nomp_get_log_no(err) == NOMP_USER_ARGS_IS_INVALID);
  err = nomp_finalize();
  nomp_assert(nomp_get_log_no(err) == NOMP_RUNTIME_NOT_INITIALIZED);

  return 0;
}
