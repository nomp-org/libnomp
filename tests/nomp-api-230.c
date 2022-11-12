#define TEST_IMPL_H "nomp-api-230-impl.h"
#include "nomp-generate-tests.h"
#undef TEST_IMPL_H

int main(int argc, char *argv[]) {
  char *backend;
  int device, platform;
  parse_input(argc, argv, &backend, &device, &platform);

  TEST_SUITE(230, backend, device, platform)

  return 0;
}
