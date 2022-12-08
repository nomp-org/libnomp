#include "nomp-test.h"
#include "nomp.h"

// Free'ing before mapping should return an error
static int test_free_before_mapping(char *backend, int platform, int device) {
  int a[10] = {0};
  int err = nomp_init(backend, platform, device);
  nomp_test_chk(err);
  err = nomp_update(a, 0, 10, sizeof(int), NOMP_FREE);
  nomp_test_assert(nomp_get_log_no(err) == NOMP_USER_MAP_OP_IS_INVALID);

  char *desc;
  err = nomp_get_log_str(&desc, err);
  int matched = match_log(
      desc, "\\[Error\\] "
            ".*libnomp\\/src\\/nomp.c:[0-9]* NOMP_FREE or NOMP_FROM can only "
            "be called on a pointer which is already on the device.");
  nomp_test_assert(matched);
  tfree(desc);

  return 0;
}

// D2H before H2D should return an error
static int test_d2h_before_h2d() {
  int a[10] = {0};
  int err = nomp_update(a, 0, 10, sizeof(int), NOMP_FROM);
  nomp_test_assert(nomp_get_log_no(err) == NOMP_USER_MAP_OP_IS_INVALID);

  char *desc;
  err = nomp_get_log_str(&desc, err);
  int matched = match_log(
      desc, "\\[Error\\] "
            ".*libnomp\\/src\\/nomp.c:[0-9]* NOMP_FREE or NOMP_FROM can only "
            "be called on a pointer which is already on the device.");
  nomp_test_assert(matched);

  err = nomp_finalize();
  nomp_test_chk(err);

  tfree(desc);
  return 0;
}

int main(int argc, char *argv[]) {
  char *backend;
  int device, platform;
  parse_input(argc, argv, &backend, &device, &platform);
  int err = 0;

  err |= SUBTEST(test_free_before_mapping, backend, platform, device);
  err |= SUBTEST(test_d2h_before_h2d);

  return err;
}
