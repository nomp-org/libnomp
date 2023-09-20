#define TEST_MAX_SIZE 40
#define TEST_IMPL_H "nomp-api-300-impl.h"
#include "nomp-generate-tests.h"
#undef TEST_IMPL_H
#undef TEST_MAX_SIZE

static int test_matrix_addition(void) {
  int err = 0;
  TEST_BUILTIN_TYPES(300_add, 40, 5)
  TEST_BUILTIN_TYPES(300_add, 16, 16)
  return err;
}

static int test_matrix_transpose(void) {
  int err = 0;
  TEST_BUILTIN_TYPES(300_transpose, 40, 5)
  TEST_BUILTIN_TYPES(300_transpose, 16, 16)
  return err;
}

static int test_matrix_matrix_multiplication(void) {
  int err = 0;
  TEST_BUILTIN_TYPES(300_mxm, 10)
  TEST_BUILTIN_TYPES(300_mxm, 40)
  return err;
}

static int test_matrix_vector_multiplication(void) {
  int err = 0;
  TEST_BUILTIN_TYPES(300_vxm, 10)
  TEST_BUILTIN_TYPES(300_vxm, 40)
  return err;
}

int main(int argc, const char *argv[]) {
  int err = nomp_init(argc, argv);
  nomp_test_chk(err);

  err |= SUBTEST(test_matrix_addition);
  err |= SUBTEST(test_matrix_transpose);
  err |= SUBTEST(test_matrix_vector_multiplication);
  err |= SUBTEST(test_matrix_matrix_multiplication);

  err |= nomp_finalize();
  nomp_test_chk(err);

  return err;
}
