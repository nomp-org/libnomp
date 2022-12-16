#define TEST_IMPL_H "nomp-api-232-impl.h"
#include "nomp-generate-tests.h"
#undef TEST_IMPL_H

int main(int argc, const char *argv[]) {
  int err = 0;
  TEST_BUILTIN_TYPES(232, argc, argv)
  return err;
}
