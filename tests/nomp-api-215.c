#define TEST_MAX_SIZE 100
#define TEST_IMPL_H "nomp-api-215-impl.h"
#include "nomp-generate-tests.h"
#undef TEST_IMPL_H
#undef TEST_MAX_SIZE

static int test_logical_ops(void) {
  int err = 0;
  TEST_BUILTIN_TYPES(215_logical_ops, 10)
  TEST_BUILTIN_TYPES(215_logical_ops, 50)
  return err;
}

static int test_ternary_ops(void) {
  int err = 0;
  TEST_BUILTIN_TYPES(215_ternary_ops, 10)
  TEST_BUILTIN_TYPES(215_ternary_ops, 50)
  return err;
}

int main(int argc, const char *argv[]) {
  int err = nomp_init(argc, argv);
  nomp_test_chk(err);

  err |= SUBTEST(test_logical_ops);
  err |= SUBTEST(test_ternary_ops);

  err |= nomp_finalize();
  nomp_test_chk(err);

  return err;
}
