#define TEST_MAX_SIZE 256
#define TEST_IMPL_H "nomp-api-350-impl.h"
#include "nomp-generate-tests.h"
#undef TEST_IMPL_H
#undef TEST_MAX_SIZE

// Test non-constant/ non-single-variable for loop bounds inside a serial loop.
static int test_for_loop_bounds(void) {
  int err = 0;
  TEST_BUILTIN_TYPES(350_for_loop_bounds, 10)
  TEST_BUILTIN_TYPES(350_for_loop_bounds, 50)
  return err;
}

int main(int argc, const char *argv[]) {
  int err = nomp_init(argc, argv);
  nomp_check(err);

  err |= SUBTEST(test_for_loop_bounds);

  err |= nomp_finalize();
  nomp_check(err);

  return err;
}
