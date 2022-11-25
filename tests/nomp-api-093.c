#include "nomp-test.h"
#include "nomp.h"
#include <limits.h>

int main(int argc, char *argv[]) {

  // Run with a valid device-id environment variable
  setenv("NOMP_DEVICE_ID", "0", 1);
  int err = nomp_init(&argc, &argv);
  nomp_chk(err);
  err = nomp_finalize();
  nomp_chk(err);
  unsetenv("NOMP_DEVICE_ID");

  return 0;
}
