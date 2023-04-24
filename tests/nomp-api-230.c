#define TEST_IMPL_H "nomp-api-230-impl.h"
#include "nomp-generate-tests.h"
#undef TEST_IMPL_H

#include "nomp-aux.h"

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
  const char *additional_args[2] = {"--nomp-function", "sem::annotate"};
  const char *new_argv[64];
  const int new_argc = argc + 2;
  for (int i = 0; i < argc; i++)
    new_argv[i] = argv[i];
  for (int i = argc; i < new_argc; i++)
    new_argv[i] = additional_args[i - argc];

  int err = nomp_init(new_argc, new_argv);
  nomp_check(err);

  err |= SUBTEST(test_matrix_addition);
  err |= SUBTEST(test_matrix_transform);

  err |= nomp_finalize();
  nomp_check(err);

  return err;
}
