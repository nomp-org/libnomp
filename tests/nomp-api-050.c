#include "nomp-test.h"
#include "nomp.h"

// Free'ing before mapping should return an error.
static int test_free_before_mapping() {
  int a[10] = {0};
  int err = nomp_update(a, 0, 10, sizeof(int), NOMP_FREE);
  nomp_test_assert(nomp_get_log_no(err) == NOMP_USER_MAP_OP_IS_INVALID);

  char *desc;
  nomp_get_log_str(&desc, err);
  int matched = match_log(
      desc, "\\[Error\\] "
            ".*libnomp\\/src\\/nomp.c:[0-9]* NOMP_FREE or NOMP_FROM can only "
            "be called on a pointer which is already on the device.");
  tfree(desc);
  nomp_test_assert(matched);

  return 0;
}

// D2H before H2D should return an error.
static int test_d2h_before_h2d() {
  int a[10] = {0};
  int err = nomp_update(a, 0, 10, sizeof(int), NOMP_FROM);
  nomp_test_assert(nomp_get_log_no(err) == NOMP_USER_MAP_OP_IS_INVALID);

  char *desc;
  nomp_get_log_str(&desc, err);
  int matched = match_log(
      desc, "\\[Error\\] "
            ".*libnomp\\/src\\/nomp.c:[0-9]* NOMP_FREE or NOMP_FROM can only "
            "be called on a pointer which is already on the device.");
  tfree(desc);
  nomp_test_assert(matched);

  return 0;
}

// Mapping H2D multiple times is not an error.
static int test_multiple_h2d_calls() {
  int a[10] = {0};
  int err = nomp_update(a, 0, 10, sizeof(int), NOMP_TO);
  nomp_test_chk(err);
  err = nomp_update(a, 0, 10, sizeof(int), NOMP_TO);
  nomp_test_chk(err);

  return 0;
}

// Mapping D2H multiple times is not an error.
static int test_multiple_d2h_calls() {
  int a[10] = {0};
  int err = nomp_update(a, 0, 10, sizeof(int), NOMP_TO);
  nomp_test_chk(err);

  err = nomp_update(a, 0, 10, sizeof(int), NOMP_FROM);
  nomp_test_chk(err);
  err = nomp_update(a, 0, 10, sizeof(int), NOMP_FROM);
  nomp_test_chk(err);

  return 0;
}

// Check D2H followed by H2D.
static int test_d2h_after_h2d() {
  int a[10] = {0};

  for (unsigned i = 0; i < 10; i++)
    a[i] = i;
  int err = nomp_update(a, 0, 10, sizeof(int), NOMP_TO);
  nomp_test_chk(err);

  for (unsigned i = 0; i < 10; i++)
    a[i] = 0;

  err = nomp_update(a, 0, 10, sizeof(int), NOMP_FROM);
  nomp_test_chk(err);

  for (unsigned i = 0; i < 10; i++)
    nomp_test_assert(a[i] == i);

  return 0;
}

// Free'ing after H2D is not an error.
static int test_free_after_h2d() {
  int a[10] = {0};
  int err = nomp_update(a, 0, 10, sizeof(int), NOMP_TO);
  nomp_test_chk(err);

  err = nomp_update(a, 0, 10, sizeof(int), NOMP_FREE);
  nomp_test_chk(err);

  return 0;
}

// Free'ing after D2H is not an error.
static int test_free_after_d2h() {
  int a[10] = {0};
  int err = nomp_update(a, 0, 10, sizeof(int), NOMP_TO);
  nomp_test_chk(err);

  err = nomp_update(a, 0, 10, sizeof(int), NOMP_FROM);
  nomp_test_chk(err);

  err = nomp_update(a, 0, 10, sizeof(int), NOMP_FREE);
  nomp_test_chk(err);

  return 0;
}

int main(int argc, const char *argv[]) {
  int err = nomp_init(argc, argv);
  nomp_test_chk(err);

  err |= SUBTEST(test_free_before_mapping);
  err |= SUBTEST(test_d2h_before_h2d);
  err |= SUBTEST(test_multiple_h2d_calls);
  err |= SUBTEST(test_multiple_d2h_calls);
  err |= SUBTEST(test_d2h_after_h2d);
  err |= SUBTEST(test_free_after_h2d);
  err |= SUBTEST(test_free_after_d2h);

  err = nomp_finalize();
  nomp_test_chk(err);

  return err;
}
