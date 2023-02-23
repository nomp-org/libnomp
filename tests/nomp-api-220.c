#define TEST_IMPL_H "nomp-api-220-impl.h"
#include "nomp-generate-tests.h"
#undef TEST_IMPL_H

static int test_vector_addition() {
  int err = 0;
  TEST_BUILTIN_TYPES(220_add, 10)
  TEST_BUILTIN_TYPES(220_add, 20)
  return err;
}

static int test_vector_multiplication() {
  int err = 0;
  TEST_BUILTIN_TYPES(220_mul, 10)
  TEST_BUILTIN_TYPES(220_mul, 20)
  return err;
}

static int test_vector_multiplication_sum() {
  int err = 0;
  TEST_BUILTIN_TYPES(220_mul_sum, 10)
  TEST_BUILTIN_TYPES(220_mul_sum, 20)
  return err;
}

static int test_vector_linear() {
  int err = 0;
  TEST_BUILTIN_TYPES(220_linear, 10)
  TEST_BUILTIN_TYPES(220_linear, 20)
  return err;
}

int main(int argc, const char *argv[]) {
  int err = nomp_init(argc, argv);
  nomp_check(err);

  err |= SUBTEST(test_vector_addition);
  err |= SUBTEST(test_vector_multiplication);
  err |= SUBTEST(test_vector_multiplication_sum);
  err |= SUBTEST(test_vector_linear);

  err |= nomp_finalize();
  nomp_check(err);

  return err;
}
