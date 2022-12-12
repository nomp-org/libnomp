#define TEST_IMPL_H "nomp-api-110-impl.h"
#include "nomp-generate-tests.h"
#undef TEST_IMPL_H

int main(int argc, const char *argv[]) {

  TEST_BUILTIN_TYPES(110, argc, argv, 0, 10)
  TEST_BUILTIN_TYPES(110, argc, argv, 5, 10)
  TEST_BUILTIN_TYPES(110, argc, argv, 2, 8)

  return 0;
}
