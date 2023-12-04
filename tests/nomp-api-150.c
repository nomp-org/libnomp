#include "nomp-test.h"

#define TEST_MAX_SIZE 32
#define TEST_IMPL_H "nomp-api-150-impl.h"
#include "nomp-generate-tests.h"
#undef TEST_IMPL_H
#undef TEST_MAX_SIZE

static int test_invalid_kernel_id(int n) {
  int err = 0;
  TEST_BUILTIN_TYPES(150_invalid_kernel_id, n);
  return err;
}

static int test_unmapped_array(int n) {
  int err = 0;
  TEST_BUILTIN_TYPES(150_unmapped_array, n);
  return err;
}

int main(int argc, const char *argv[]) {
  nomp_test_check(nomp_init(argc, argv));

  const int n = 20;
  int err = 0;
  err |= SUBTEST(test_invalid_kernel_id, n);
  err |= SUBTEST(test_unmapped_array, n);

  nomp_test_check(nomp_finalize());

  return err;
}
