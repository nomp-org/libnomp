#define TEST_IMPL_H "nomp-api-240-impl.h"
#include "nomp-generate-tests.h"
#undef TEST_IMPL_H

int main(int argc, const char *argv[]) {

  TEST_BUILTIN_TYPES(240, argc, argv)

  return 0;
}
