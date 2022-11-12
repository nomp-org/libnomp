#define TEST_IMPL_H "nomp-api-230-impl.h"
#include "nomp-generate-tests.h"
#undef TEST_IMPL_H

int main(int argc, char *argv[]) {
  char *backend;
  int device, platform;
  parse_input(argc, argv, &backend, &device, &platform);

  nomp_api_230_int(backend, device, platform);
  nomp_api_230_long(backend, device, platform);
  nomp_api_230_unsigned(backend, device, platform);
  nomp_api_230_unsigned_long(backend, device, platform);
  nomp_api_230_float(backend, device, platform);
  nomp_api_230_double(backend, device, platform);

  return 0;
}
