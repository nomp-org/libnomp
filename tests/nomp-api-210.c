#define TEST_INT_ONLY

#define TEST_MAX_SIZE 100
#define TEST_IMPL_H "nomp-api-210-impl.h"
#include "nomp-generate-tests.h"
#undef TEST_IMPL_H
#undef TEST_MAX_SIZE

static int test_bitwise_and() {
  int err = 0;
  TEST_BUILTIN_TYPES(210_bitwise_and, 10)
  TEST_BUILTIN_TYPES(210_bitwise_and, 50)
  TEST_BUILTIN_TYPES(210_bitwise_and, 70)
  return err;
}

static int test_bitwise_or() {
  int err = 0;
  TEST_BUILTIN_TYPES(210_bitwise_or, 10)
  TEST_BUILTIN_TYPES(210_bitwise_or, 50)
  TEST_BUILTIN_TYPES(210_bitwise_or, 70)
  return err;
}

static int test_bitwise_xor() {
  int err = 0;
  TEST_BUILTIN_TYPES(210_bitwise_xor, 10)
  TEST_BUILTIN_TYPES(210_bitwise_xor, 50)
  TEST_BUILTIN_TYPES(210_bitwise_xor, 70)
  return err;
}

static int test_bitwise_left_shift() {
  int err = 0;
  TEST_BUILTIN_TYPES(210_bitwise_left_shift, 10)
  TEST_BUILTIN_TYPES(210_bitwise_left_shift, 50)
  TEST_BUILTIN_TYPES(210_bitwise_left_shift, 70)
  return err;
}

static int test_bitwise_right_shift() {
  int err = 0;
  TEST_BUILTIN_TYPES(210_bitwise_right_shift, 10)
  TEST_BUILTIN_TYPES(210_bitwise_right_shift, 50)
  TEST_BUILTIN_TYPES(210_bitwise_right_shift, 70)
  return err;
}

static int test_bitwise_complement() {
  int err = 0;
  TEST_BUILTIN_TYPES(210_bitwise_complement, 10)
  TEST_BUILTIN_TYPES(210_bitwise_complement, 50)
  TEST_BUILTIN_TYPES(210_bitwise_complement, 70)
  return err;
}

int main(int argc, const char *argv[]) {
  int err = nomp_init(argc, argv);
  nomp_check(err);

  err |= SUBTEST(test_bitwise_and);
  err |= SUBTEST(test_bitwise_or);
  err |= SUBTEST(test_bitwise_xor);
  err |= SUBTEST(test_bitwise_left_shift);
  err |= SUBTEST(test_bitwise_right_shift);
  err |= SUBTEST(test_bitwise_complement);

  err |= nomp_finalize();
  nomp_check(err);

  return err;
}

#undef TEST_INT_ONLY
