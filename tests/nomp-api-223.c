#define TEST_IMPL_H "nomp-api-223-impl.h"
#include "nomp-generate-tests.h"
#undef TEST_IMPL_H

int main(int argc, const char *argv[]) {

  int err = nomp_init(argc, argv);
  nomp_test_chk(err);

  TEST_BUILTIN_TYPES(223_add, 10)
  TEST_BUILTIN_TYPES(223_add, 20)
  TEST_BUILTIN_TYPES(223_sub, 10)
  TEST_BUILTIN_TYPES(223_sub, 20)
  TEST_BUILTIN_TYPES(223_mul, 20)
  TEST_BUILTIN_TYPES(223_mul, 20)

  err = nomp_finalize();
  nomp_test_chk(err);

  return 0;
}
