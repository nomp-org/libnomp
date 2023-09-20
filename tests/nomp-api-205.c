#define TEST_MAX_SIZE 100
#define TEST_IMPL_H "nomp-api-205-impl.h"
#include "nomp-generate-tests.h"
#undef TEST_IMPL_H
#undef TEST_MAX_SIZE

static int test_vector_addition(void) {
  int err = 0;
  TEST_BUILTIN_TYPES(205_add, 10)
  TEST_BUILTIN_TYPES(205_add, 70)
  return err;
}

static int test_vector_multiplication(void) {
  int err = 0;
  TEST_BUILTIN_TYPES(205_mul, 10)
  TEST_BUILTIN_TYPES(205_mul, 70)
  return err;
}

static int test_vector_multiplication_sum(void) {
  int err = 0;
  TEST_BUILTIN_TYPES(205_mul_sum, 10)
  TEST_BUILTIN_TYPES(205_mul_sum, 70)
  return err;
}

static int test_vector_linear(void) {
  int err = 0;
  TEST_BUILTIN_TYPES(205_linear, 10)
  TEST_BUILTIN_TYPES(205_linear, 70)
  return err;
}

int main(int argc, const char *argv[]) {
  int err = nomp_init(argc, argv);
  nomp_test_check(err);

  err |= SUBTEST(test_vector_addition);
  err |= SUBTEST(test_vector_multiplication);
  err |= SUBTEST(test_vector_multiplication_sum);
  err |= SUBTEST(test_vector_linear);

  err |= nomp_finalize();
  nomp_test_check(err);

  return err;
}
