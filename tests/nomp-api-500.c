#define TEST_IMPL_H "nomp-api-500-impl.h"
#include "nomp-generate-tests.h"
#undef TEST_IMPL_H

static int test_sum_reduction() {
  int err = 0;
  TEST_BUILTIN_TYPES(500_sum, 10);
  return err;
}

static int test_sum_condition_reduction() {
  int err = 0;
  TEST_BUILTIN_TYPES(500_condition, 10);
  return err;
}

static int test_mxm() {
  int err = 0;
  TEST_BUILTIN_TYPES(500_mxm, 10);
  return err;
}

static int test_vxm() {
  int err = 0;
  TEST_BUILTIN_TYPES(500_vxm, 10);
  return err;
}

int main(int argc, const char *argv[]) {
  nomp_check(nomp_init(argc, argv));

  int err = 0;
  err |= SUBTEST(test_sum_reduction);
  /// TODO: Fix the errors of the following kernels
  //  err |= SUBTEST(test_sum_condition_reduction);
  //  err |= SUBTEST(test_mxm);
  //  err |= SUBTEST(test_vxm);

  nomp_check(nomp_finalize());

  return err;
}
