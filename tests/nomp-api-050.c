#include "nomp-test.h"
#include "nomp.h"

#define TEST_MAX_SIZE 50
#define TEST_IMPL_H "nomp-api-050-impl.h"
#include "nomp-generate-tests.h"
#undef TEST_IMPL_H
#undef TEST_MAX_SIZE

// Free'ing before mapping should return an error.
static int test_free_before_mapping(void) {
  int err = 0;
  TEST_BUILTIN_TYPES(050_free_before_mapping, 0, 10);
  TEST_BUILTIN_TYPES(050_free_before_mapping, 5, 10);
  TEST_BUILTIN_TYPES(050_free_before_mapping, 2, 8);
  return err;
}

// D2H before H2D should return an error.
static int test_d2h_before_h2d(void) {
  int err = 0;
  TEST_BUILTIN_TYPES(050_d2h_before_h2d, 0, 10);
  TEST_BUILTIN_TYPES(050_d2h_before_h2d, 5, 10);
  TEST_BUILTIN_TYPES(050_d2h_before_h2d, 2, 8);
  return err;
}

// Mapping H2D multiple times is not an error.
static int test_multiple_h2d_calls(void) {
  int err = 0;
  TEST_BUILTIN_TYPES(050_multiple_h2d_calls, 0, 10);
  TEST_BUILTIN_TYPES(050_multiple_h2d_calls, 5, 10);
  TEST_BUILTIN_TYPES(050_multiple_h2d_calls, 2, 8);
  return err;
}

// Mapping D2H multiple times is not an error.
static int test_multiple_d2h_calls(void) {
  int err = 0;
  TEST_BUILTIN_TYPES(050_multiple_d2h_calls, 0, 10);
  TEST_BUILTIN_TYPES(050_multiple_d2h_calls, 5, 10);
  TEST_BUILTIN_TYPES(050_multiple_d2h_calls, 2, 8);
  return err;
}

// Check D2H followed by H2D.
static int test_d2h_after_h2d(void) {
  int err = 0;
  TEST_BUILTIN_TYPES(050_d2h_after_h2d, 0, 10);
  TEST_BUILTIN_TYPES(050_d2h_after_h2d, 5, 10);
  TEST_BUILTIN_TYPES(050_d2h_after_h2d, 2, 8);
  return err;
}

// Free'ing after H2D is not an error.
static int test_free_after_h2d(void) {
  int err = 0;
  TEST_BUILTIN_TYPES(050_free_after_h2d, 0, 10);
  TEST_BUILTIN_TYPES(050_free_after_h2d, 5, 10);
  TEST_BUILTIN_TYPES(050_free_after_h2d, 2, 8);
  return err;
}

// Free'ing after D2H is not an error.
static int test_free_after_d2h(void) {
  int err = 0;
  TEST_BUILTIN_TYPES(050_free_after_d2h, 0, 10);
  TEST_BUILTIN_TYPES(050_free_after_d2h, 5, 10);
  TEST_BUILTIN_TYPES(050_free_after_d2h, 2, 8);
  return err;
}

static int test_in_range_d2h(void) {
  int err = 0;
  TEST_BUILTIN_TYPES(050_in_range_d2h, 0, 20, 5, 15);
  TEST_BUILTIN_TYPES(050_in_range_d2h, 0, 20, 0, 15);
  TEST_BUILTIN_TYPES(050_in_range_d2h, 0, 20, 5, 20);
  return err;
}

static int test_in_range_h2d(void) {
  int err = 0;
  TEST_BUILTIN_TYPES(050_in_range_h2d, 0, 20, 5, 15);
  TEST_BUILTIN_TYPES(050_in_range_h2d, 0, 20, 1, 15);
  TEST_BUILTIN_TYPES(050_in_range_h2d, 0, 20, 5, 19);
  return err;
}

static int test_dynamic_data_type(void) {
  int err = 0;
  TEST_BUILTIN_TYPES(050_dynamic_data_type, 16);
  TEST_BUILTIN_TYPES(050_dynamic_data_type, 32);
  return err;
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
  err |= SUBTEST(test_in_range_d2h);
  err |= SUBTEST(test_in_range_h2d);
  err |= SUBTEST(test_dynamic_data_type);

  err |= nomp_finalize();
  nomp_test_chk(err);

  return err;
}
