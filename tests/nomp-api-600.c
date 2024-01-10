#define TEST_MAX_SIZE 100
#define TEST_IMPL_H "nomp-api-600-impl.h"
#include "nomp-generate-tests.h"
#undef TEST_IMPL_H
#undef TEST_MAX_SIZE

static int test_vector_addition(void) {
  int err = 0;
  TEST_BUILTIN_TYPES(600_add, 10)
  TEST_BUILTIN_TYPES(600_add, 50)
  return err;
}

int main(int argc, const char *argv[]) {
  nomp_test_check(nomp_init(argc, argv));

  int err = 0;
  err |= SUBTEST(test_vector_addition);
  nomp_test_check(nomp_finalize());

  return err;
}
