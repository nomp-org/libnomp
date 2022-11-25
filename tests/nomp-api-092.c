#include "nomp-test.h"
#include "nomp.h"
#include <limits.h>

int main(int argc, char *argv[]) {

  // If both are invalid should return an error
  unsetenv("NOMP_BACKEND");
  setenv("NOMP_PLATFORM_ID", "invalid", 1);
  int err = nomp_init(&argc, &argv);
  nomp_assert(nomp_get_log_no(err) == NOMP_USER_PLATFORM_IS_INVALID);
  err = nomp_finalize();
  nomp_assert(nomp_get_log_no(err) == NOMP_RUNTIME_NOT_INITIALIZED);

  return 0;
}
