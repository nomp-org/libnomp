#define TEST_INT_ONLY

#define TEST_MAX_SIZE 100
#define TEST_IMPL_H "nomp-api-220-impl.h"
#include "nomp-generate-tests.h"
#undef TEST_IMPL_H
#undef TEST_MAX_SIZE

static int test_bitwise_and(void) {
  int err = 0;
  TEST_BUILTIN_TYPES(220_bitwise_and, 10)
  TEST_BUILTIN_TYPES(220_bitwise_and, 50)
  TEST_BUILTIN_TYPES(220_bitwise_and, 70)
  return err;
}

static int test_bitwise_or(void) {
  int err = 0;
  TEST_BUILTIN_TYPES(220_bitwise_or, 10)
  TEST_BUILTIN_TYPES(220_bitwise_or, 50)
  TEST_BUILTIN_TYPES(220_bitwise_or, 70)
  return err;
}

static int test_bitwise_xor(void) {
  int err = 0;
  TEST_BUILTIN_TYPES(220_bitwise_xor, 10)
  TEST_BUILTIN_TYPES(220_bitwise_xor, 50)
  TEST_BUILTIN_TYPES(220_bitwise_xor, 70)
  return err;
}

static int test_bitwise_left_shift(void) {
  int err = 0;
  TEST_BUILTIN_TYPES(220_bitwise_left_shift, 10)
  TEST_BUILTIN_TYPES(220_bitwise_left_shift, 50)
  TEST_BUILTIN_TYPES(220_bitwise_left_shift, 70)
  return err;
}

static int test_bitwise_right_shift(void) {
  int err = 0;
  TEST_BUILTIN_TYPES(220_bitwise_right_shift, 10)
  TEST_BUILTIN_TYPES(220_bitwise_right_shift, 50)
  TEST_BUILTIN_TYPES(220_bitwise_right_shift, 70)
  return err;
}

static int test_bitwise_complement(void) {
  int err = 0;
  TEST_BUILTIN_TYPES(220_bitwise_complement, 10)
  TEST_BUILTIN_TYPES(220_bitwise_complement, 50)
  TEST_BUILTIN_TYPES(220_bitwise_complement, 70)
  return err;
}

int main(int argc, const char *argv[]) {
  nomp_test_check(nomp_init(argc, argv));

  int err = 0;
  err |= SUBTEST(test_bitwise_and);
  err |= SUBTEST(test_bitwise_or);
  err |= SUBTEST(test_bitwise_xor);
  err |= SUBTEST(test_bitwise_left_shift);
  err |= SUBTEST(test_bitwise_right_shift);
  err |= SUBTEST(test_bitwise_complement);

  nomp_test_check(nomp_finalize());

  return err;
}

#undef TEST_INT_ONLY
