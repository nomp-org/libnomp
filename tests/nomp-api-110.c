#define TEST_IMPL_H "nomp-api-110-impl.h"
#include "nomp-generate-tests.h"
#undef TEST_IMPL_H

int main(int argc, char *argv[]) {
  char *backend;
  int device, platform;
  parse_input(argc, argv, &backend, &device, &platform);

  TEST_SUITE(110, backend, device, platform, 0, 10)
  TEST_SUITE(110, backend, device, platform, 5, 10)
  TEST_SUITE(110, backend, device, platform, 2, 8)

  return 0;
}
