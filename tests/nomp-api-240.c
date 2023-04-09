#define TEST_IMPL_H "nomp-api-240-impl.h"
#include "nomp-generate-tests.h"
#undef TEST_IMPL_H

// Test break statement inside a serial loop.
static int test_break() {
  int err = 0;
  TEST_BUILTIN_TYPES(240_break, 10)
  return err;
}

// Test continue statement inside a serial loop.
static int test_continue() {
  int err = 0;
  TEST_BUILTIN_TYPES(240_continue, 10)
  return err;
}

// Test logical operators inside a serial loop.
static int test_logical_ops() {
  int err = 0;
  TEST_BUILTIN_TYPES(240_logical_ops, 10)
  return err;
}

int main(int argc, const char *argv[]) {
  int err = nomp_init(argc, argv);
  nomp_check(err);

  err |= SUBTEST(test_break);
  err |= SUBTEST(test_continue);
  err |= SUBTEST(test_logical_ops);

  err |= nomp_finalize();
  nomp_check(err);

  return err;
}
