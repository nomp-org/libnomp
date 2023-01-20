#define TEST_IMPL_H "nomp-api-100-impl.h"
#include "nomp-generate-tests.h"
#undef TEST_IMPL_H

int main(int argc, const char *argv[]) {
  int err = nomp_init(argc, argv);
  nomp_test_chk(err);

  TEST_BUILTIN_TYPES(100, 0, 10)
  TEST_BUILTIN_TYPES(100, 5, 10)
  TEST_BUILTIN_TYPES(100, 2, 8)

  err = nomp_finalize();
  nomp_test_chk(err);

  return err;
}
