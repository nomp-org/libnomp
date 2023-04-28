#define TEST_IMPL_H "nomp-api-241-impl.h"
#include "nomp-generate-tests.h"
#undef TEST_IMPL_H

static int test_logical_ops() {
  int err = 0;
  TEST_BUILTIN_TYPES(241_logical_ops, 10)
  return err;
}

static int test_ternary_ops() {
  int err = 0;
  TEST_BUILTIN_TYPES(241_ternary_ops, 10)
  return err;
}

int main(int argc, const char *argv[]) {
  int err = nomp_init(argc, argv);
  nomp_check(err);

  err |= SUBTEST(test_logical_ops);
  err |= SUBTEST(test_ternary_ops);

  err |= nomp_finalize();
  nomp_check(err);

  return err;
}
