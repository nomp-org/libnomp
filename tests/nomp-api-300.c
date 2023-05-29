#define TEST_MAX_SIZE 256
#define TEST_IMPL_H "nomp-api-300-impl.h"
#include "nomp-generate-tests.h"
#undef TEST_IMPL_H
#undef TEST_MAX_SIZE

static int test_matrix_addition() {
  int err = 0;
  TEST_BUILTIN_TYPES(300_add, 50, 5)
  TEST_BUILTIN_TYPES(300_add, 16, 16)
  return err;
}

static int test_matrix_transform() {
  int err = 0;
  TEST_BUILTIN_TYPES(300_transform, 50, 5)
  TEST_BUILTIN_TYPES(300_transform, 16, 16)
  return err;
}

int main(int argc, const char *argv[]) {
  int err = nomp_init(argc, argv);
  nomp_check(err);

  err |= SUBTEST(test_matrix_addition);
  err |= SUBTEST(test_matrix_transform);

  err |= nomp_finalize();
  nomp_check(err);

  return err;
}
