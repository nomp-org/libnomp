#include "nomp-test.h"
#include "nomp.h"

int main(int argc, char *argv[]) {
  char *backend;
  int device, platform;
  parse_input(argc, argv, &backend, &device, &platform);
  int a[10] = {0};

  int err = nomp_init(backend, platform, device);

  // Free'ing before mapping should return an error
  err = nomp_update(a, 0, 10, sizeof(int), NOMP_FREE);
  nomp_assert(nomp_get_log_no(err) == NOMP_USER_MAP_OP_IS_INVALID);

  char *desc;
  err = nomp_get_log_str(&desc, err);
  int matched = match_log(
      desc, "\\[Error\\] "
            ".*libnomp\\/src\\/nomp.c:[0-9]* NOMP_FREE or NOMP_FROM can only "
            "be called on a pointer which is already on the device.");
  nomp_assert(matched);
  tfree(desc);

  // D2H before H2D should return an error
  err = nomp_update(a, 0, 10, sizeof(int), NOMP_FROM);
  nomp_assert(nomp_get_log_no(err) == NOMP_USER_MAP_OP_IS_INVALID);

  err = nomp_get_log_str(&desc, err);
  matched = match_log(
      desc, "\\[Error\\] "
            ".*libnomp\\/src\\/nomp.c:[0-9]* NOMP_FREE or NOMP_FROM can only "
            "be called on a pointer which is already on the device.");
  nomp_assert(matched);

  err = nomp_finalize();
  nomp_chk(err);

  tfree(desc);

  return 0;
}
