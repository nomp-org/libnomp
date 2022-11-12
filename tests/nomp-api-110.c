#include "nomp.h"

#define TEST_IMPL_H "nomp-api-110-impl.h"
#include "nomp-generate-tests.h"
#undef TEST_IMPL_H

int main(int argc, char *argv[]) {
  char *backend;
  int device, platform;
  parse_input(argc, argv, &backend, &device, &platform);

  nomp_api_110_int(backend, device, platform, 0, 10);
  nomp_api_110_long(backend, device, platform, 0, 10);
  nomp_api_110_unsigned(backend, device, platform, 0, 10);
  nomp_api_110_unsigned_long(backend, device, platform, 0, 10);
  nomp_api_110_double(backend, device, platform, 0, 10);
  nomp_api_110_float(backend, device, platform, 0, 10);

  nomp_api_110_int(backend, device, platform, 5, 10);
  nomp_api_110_long(backend, device, platform, 5, 10);
  nomp_api_110_unsigned(backend, device, platform, 5, 10);
  nomp_api_110_unsigned_long(backend, device, platform, 5, 10);
  nomp_api_110_double(backend, device, platform, 5, 10);
  nomp_api_110_float(backend, device, platform, 5, 10);

  nomp_api_110_int(backend, device, platform, 2, 8);
  nomp_api_110_long(backend, device, platform, 2, 8);
  nomp_api_110_unsigned(backend, device, platform, 2, 8);
  nomp_api_110_unsigned_long(backend, device, platform, 2, 8);
  nomp_api_110_double(backend, device, platform, 2, 8);
  nomp_api_110_float(backend, device, platform, 2, 8);

  return 0;
}
