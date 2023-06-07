#define TEST_IMPL_H "nomp-api-500-impl.h"
#include "nomp-generate-tests.h"
#undef TEST_IMPL_H

static int test_sum() {
  int err = 0;
  TEST_BUILTIN_TYPES(500_sum_const, 10);
  TEST_BUILTIN_TYPES(500_sum_const, 50);
  TEST_BUILTIN_TYPES(500_sum_var, 10);
  TEST_BUILTIN_TYPES(500_sum_var, 50);
  TEST_BUILTIN_TYPES(500_sum_array, 10);
  TEST_BUILTIN_TYPES(500_sum_array, 50);
  return err;
}

static int test_sum_condition() {
  int err = 0;
  TEST_BUILTIN_TYPES(500_condition, 10);
  return err;
}

static int test_dot() {
  int err = 0;
  TEST_BUILTIN_TYPES(500_dot, 10);
  TEST_BUILTIN_TYPES(500_dot, 50);
  return err;
}

int main(int argc, const char *argv[]) {
  nomp_check(nomp_init(argc, argv));

  int err = 0;
  err |= SUBTEST(test_sum);
  err |= SUBTEST(test_dot);
  // FIXME: Fix the errors of the following kernels
  // err |= SUBTEST(test_sum_condition);

  nomp_check(nomp_finalize());

  return err;
}
