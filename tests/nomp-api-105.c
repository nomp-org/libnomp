#include "nomp-test.h"

#define TEST_IMPL_H "nomp-api-105-impl.h"
#include "nomp-generate-tests.h"
#undef TEST_IMPL_H

// Calling nomp_jit() with valid functions should not return an error.
static int test_valid_clauses(void) {
  int err = 0;
  TEST_BUILTIN_TYPES(105_valid, );
  return err;
}

// Calling nomp_jit() with invalid functions should return an error.
static int test_invalid_clauses(void) {
  int err = 0;
  TEST_BUILTIN_TYPES(105_invalid, );
  return err;
}

int main(int argc, const char *argv[]) {
  int err = nomp_init(argc, argv);
  nomp_test_chk(err);

  err |= SUBTEST(test_valid_clauses);
  err |= SUBTEST(test_invalid_clauses);

  err |= nomp_finalize();
  nomp_test_chk(err);

  return err;
}
