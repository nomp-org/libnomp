#include "nomp-test.h"

#define TEST_IMPL_H "nomp-api-200-impl.h"
#include "nomp-generate-tests.h"
#undef TEST_IMPL_H

// Calling nomp_jit() with valid functions should not return an error.
static int test_valid_clauses() {
  int err = 0;
  const char *clauses[4] = {"transform", "nomp-api-200", "transform", 0};
  TEST_BUILTIN_TYPES(200, clauses)
  return err;
}

// Calling nomp_jit() with invalid functions should return an error.
static int test_invalid_clauses() {
  int err = 0;
  const char *clauses[4] = {"transform", "invalid-file", "invalid_func", 0};
  TEST_BUILTIN_TYPES(200_err, clauses)
  return err;
}

int main(int argc, const char *argv[]) {
  int err = nomp_init(argc, argv);
  nomp_chk(err);

  err |= SUBTEST(test_valid_clauses);
  err |= SUBTEST(test_invalid_clauses);

  err |= nomp_finalize();
  nomp_chk(err);

  return err;
}
