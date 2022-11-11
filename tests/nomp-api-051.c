#include "nomp-test.h"
#include "nomp.h"

int main(int argc, char *argv[]) {
  char *backend = argc > 1 ? argv[1] : "opencl";
  int device = argc > 2 ? atoi(argv[2]) : 0;
  int platform = argc > 3 ? atoi(argv[3]) : 0;
  int a[10] = {0};

  int err = nomp_init(backend, platform, device);

  // Free'ing before mapping should return an error
  err = nomp_update(a, 0, 10, sizeof(int), NOMP_FREE);
  nomp_assert(nomp_get_log_no(err) == NOMP_USER_MAP_PTR_NOT_VALID);

  char *desc;
  err = nomp_get_log_str(&desc, err);
  int matched = match_log(desc, "\\[Error\\] "
                                ".*libnomp\\/src\\/nomp.c:[0-9]* "
                                "Invalid map pointer operation 8.");
  nomp_assert(matched);

  // D2H before H2D should return an error
  err = nomp_update(a, 0, 10, sizeof(int), NOMP_FROM);
  nomp_assert(nomp_get_log_no(err) == NOMP_USER_MAP_PTR_NOT_VALID);

  err = nomp_get_log_str(&desc, err);
  matched = match_log(desc, "\\[Error\\] "
                            ".*libnomp\\/src\\/nomp.c:[0-9]* "
                            "Invalid map pointer operation 4.");
  nomp_assert(matched);

  err = nomp_finalize();
  nomp_chk(err);

  tfree(desc);

  return 0;
}
