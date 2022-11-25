#include "nomp-test.h"
#include "nomp.h"
#include <limits.h>

int main(int argc, char *argv[]) {

  // Environment variable has higher priority
  setenv("NOMP_BACKEND", "opencl", 1);
  int err = nomp_init(&argc, &argv);
  nomp_chk(err);
  err = nomp_finalize();
  nomp_chk(err);

  // Environment variable case does not matter
  setenv("NOMP_BACKEND", "oPenCl", 1);
  err = nomp_init(&argc, &argv);
  nomp_chk(err);
  err = nomp_finalize();
  nomp_chk(err);
  unsetenv("NOMP_BACKEND");

  // For invalid platform-id environment variable, passed value is used
  setenv("NOMP_PLATFORM_ID", "invalid", 1);
  err = nomp_init(&argc, &argv);
  nomp_chk(err);
  err = nomp_finalize();
  nomp_chk(err);

  return 0;
}
