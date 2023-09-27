#define TEST_MAX_SIZE 100
#define TEST_IMPL_H "nomp-api-200-impl.h"
#include "nomp-generate-tests.h"
#undef TEST_IMPL_H
#undef TEST_MAX_SIZE

static int test_vector_addition(void) {
  int err = 0;
  TEST_BUILTIN_TYPES(200_add, 10)
  TEST_BUILTIN_TYPES(200_add, 50)
  return err;
}

static int test_vector_subtraction(void) {
  int err = 0;
  TEST_BUILTIN_TYPES(200_sub, 10)
  TEST_BUILTIN_TYPES(200_sub, 50)
  return err;
}

static int test_vector_multiplication1(void) {
  int err = 0;
  TEST_BUILTIN_TYPES(200_mul1, 10)
  TEST_BUILTIN_TYPES(200_mul1, 50)
  return err;
}

static int test_vector_multiplication2(void) {
  int err = 0;
  TEST_BUILTIN_TYPES(200_mul2, 10)
  TEST_BUILTIN_TYPES(200_mul2, 50)
  return err;
}

static int test_vector_square_sum(void) {
  int err = 0;
  TEST_BUILTIN_TYPES(200_square, 10)
  TEST_BUILTIN_TYPES(200_square, 50)
  return err;
}

static int test_vector_saxpy(void) {
  int err = 0;
  TEST_BUILTIN_TYPES(200_saxpy, 10)
  TEST_BUILTIN_TYPES(200_saxpy, 50)
  return err;
}

int main(int argc, const char *argv[]) {
  int err = nomp_init(argc, argv);
  nomp_test_check(err);

  err |= SUBTEST(test_vector_addition);
  err |= SUBTEST(test_vector_subtraction);
  err |= SUBTEST(test_vector_multiplication1);
  err |= SUBTEST(test_vector_multiplication2);
  err |= SUBTEST(test_vector_square_sum);
  err |= SUBTEST(test_vector_saxpy);

  err |= nomp_finalize();
  nomp_test_check(err);

  return err;
}
