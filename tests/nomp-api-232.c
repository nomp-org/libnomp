#include "nomp.h"

#define TEST_IMPL_H "nomp-api-232-impl.h"
#include "nomp-generate-tests.h"
#undef TEST_IMPL_H

int main(int argc, char *argv[]) {
  char *backend;
  int device, platform;
  parse_input(argc, argv, &backend, &device, &platform);

  nomp_api_232_int(backend, device, platform);
  nomp_api_232_long(backend, device, platform);
  nomp_api_232_unsigned(backend, device, platform);
  nomp_api_232_unsigned_long(backend, device, platform);
  nomp_api_232_float(backend, device, platform);
  nomp_api_232_double(backend, device, platform);

  return 0;
}
