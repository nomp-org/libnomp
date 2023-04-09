#define TEST_IMPL_H "nomp-api-250-impl.h"
#include "nomp-generate-tests.h"
#undef TEST_IMPL_H

static int test_bitwise_and_op() {
  int err = 0;
  TEST_BUILTIN_TYPES(250_bitwise_and_op, 10)
  return err;
}

static int test_bitwise_or_op() {
  int err = 0;
  TEST_BUILTIN_TYPES(250_bitwise_or_op, 10)
  return err;
}

static int test_bitwise_xor_op() {
  int err = 0;
  TEST_BUILTIN_TYPES(250_bitwise_xor_op, 10)
  return err;
}

static int test_bitwise_left_shift_op() {
  int err = 0;
  TEST_BUILTIN_TYPES(250_bitwise_left_shift_op, 10)
  return err;
}

static int test_bitwise_right_shift_op() {
  int err = 0;
  TEST_BUILTIN_TYPES(250_bitwise_right_shift_op, 10)
  return err;
}

static int test_bitwise_complement_op() {
  int err = 0;
  TEST_BUILTIN_TYPES(250_bitwise_complement_op, 10)
  return err;
}

int main(int argc, const char *argv[]) {
  int err = nomp_init(argc, argv);
  nomp_check(err);

  err |= SUBTEST(test_bitwise_and_op);
  err |= SUBTEST(test_bitwise_or_op);
  err |= SUBTEST(test_bitwise_xor_op);
  err |= SUBTEST(test_bitwise_left_shift_op);
  err |= SUBTEST(test_bitwise_right_shift_op);
  err |= SUBTEST(test_bitwise_complement_op);

  err |= nomp_finalize();
  nomp_check(err);

  return err;
}
