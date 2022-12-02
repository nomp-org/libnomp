#define TEST_IMPL_H "nomp-api-231-impl.h"
#include "nomp-generate-tests.h"
#undef TEST_IMPL_H

int main(int argc, char *argv[]) {
  char *backend;
  int device, platform;
  parse_input(argc, argv, &backend, &device, &platform);

  TEST_BUILTIN_TYPES(231, backend, device, platform)

  return 0;
}
