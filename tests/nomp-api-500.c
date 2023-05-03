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

int main(int argc, const char *argv[]) {
  nomp_check(nomp_init(argc, argv));

  int err = 0;
  err |= SUBTEST(test_sum_reduction);
  /// TODO: Fix the errors regarding the kernel
  //  err |= SUBTEST(test_sum_condition_reduction);

  nomp_check(nomp_finalize());

  return err;
}
