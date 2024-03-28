#define TEST_MAX_SIZE 100
#define TEST_IMPL_H   "nomp-api-240-impl.h"
#include "nomp-generate-tests.h"
#undef TEST_IMPL_H
#undef TEST_MAX_SIZE

// Test break statement inside a serial loop.
static int test_break(void) {
  int err = 0;
  TEST_BUILTIN_TYPES(240_break, 10)
  TEST_BUILTIN_TYPES(240_break, 50)
  return err;
}

// Test continue statement inside a serial loop.
static int test_continue(void) {
  int err = 0;
  TEST_BUILTIN_TYPES(240_continue, 10)
  TEST_BUILTIN_TYPES(240_continue, 50)
  return err;
}

int main(int argc, const char *argv[]) {
  nomp_test_check(nomp_init(argc, argv));

  int err = 0;
  err |= SUBTEST(test_break);
  err |= SUBTEST(test_continue);

  nomp_test_check(nomp_finalize());

  return err;
}
