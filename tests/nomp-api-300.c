#define TEST_IMPL_H "nomp-api-300-impl.h"
#include "nomp-generate-tests.h"
#undef TEST_IMPL_H

int main(int argc, const char *argv[]) {
  int err = nomp_init(argc, argv);
  nomp_test_chk(err);

  TEST_BUILTIN_TYPES(300, 3, 2)
  TEST_BUILTIN_TYPES(300, 10, 10)

  err = nomp_finalize();
  nomp_test_chk(err);

  return err;
}
