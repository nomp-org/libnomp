#define TEST_MAX_SIZE 1024
#define TEST_IMPL_H   "nomp-api-400-impl.h"
#include "nomp-generate-tests.h"
#undef TEST_IMPL_H
#undef TEST_MAX_SIZE

static int test_fixed_size_1d_array(void) {
  int err = 0;
  TEST_BUILTIN_TYPES(400_fixed_size_1d_array, 16);
  return err;
}

static int test_variable_length_1d_array(void) {
  int err = 0;
  TEST_BUILTIN_TYPES(400_dynamic_1d_array, 16, 16);
  TEST_BUILTIN_TYPES(400_dynamic_1d_array, 16, 32);
  return err;
}

static int test_fixed_size_2d_array(void) {
  int err = 0;
  TEST_BUILTIN_TYPES(400_fixed_size_2d_array, 16);
  return err;
}

static int test_variable_length_2d_array(void) {
  int err = 0;
  TEST_BUILTIN_TYPES(400_dynamic_2d_array, 16, 16);
  TEST_BUILTIN_TYPES(400_dynamic_2d_array, 16, 32);
  return err;
}

static int test_fixed_size_3d_array(void) {
  int err = 0;
  TEST_BUILTIN_TYPES(400_fixed_size_3d_array, 16);
  return err;
}

static int test_variable_length_3d_array(void) {
  int err = 0;
  TEST_BUILTIN_TYPES(400_dynamic_3d_array, 16, 8);
  TEST_BUILTIN_TYPES(400_dynamic_3d_array, 16, 4);
  return err;
}

int main(int argc, const char **argv) {
  nomp_test_check(nomp_init(argc, argv));

  int err = 0;
  err |= SUBTEST(test_fixed_size_1d_array);
  err |= SUBTEST(test_variable_length_1d_array);
  err |= SUBTEST(test_fixed_size_2d_array);
  err |= SUBTEST(test_variable_length_2d_array);
  err |= SUBTEST(test_fixed_size_3d_array);
  err |= SUBTEST(test_variable_length_3d_array);

  nomp_test_check(nomp_finalize());

  return err;
}
