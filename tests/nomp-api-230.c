#define TEST_IMPL_H "nomp-api-230-impl.h"
#include "nomp-generate-tests.h"
#undef TEST_IMPL_H

static int test_matrix_addition() {
  int err = 0;
  TEST_BUILTIN_TYPES(230_add, 32, 4)
  TEST_BUILTIN_TYPES(230_add, 10, 10)
  return err;
}

static int test_matrix_transform() {
  int err = 0;
  TEST_BUILTIN_TYPES(230_transform, 32, 4)
  TEST_BUILTIN_TYPES(230_transform, 10, 10)
  return err;
}

int main(int argc, const char *argv[]) {
  const char *args[] = {"--nomp-backend",  "opencl",  "--nomp-device", "0",
                        "--nomp-platform", "0",       "--nomp-script", "sem",
                        "--nomp-function", "annotate"};
  int err = nomp_init(10, args);
  nomp_check(err);

  err |= SUBTEST(test_matrix_addition);
  err |= SUBTEST(test_matrix_transform);

  err |= nomp_finalize();
  nomp_check(err);

  return err;
}
