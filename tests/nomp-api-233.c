#include "nomp.h"

#define TEST_IMPL_H "nomp-api-233-impl.h"
#include "nomp-generate-tests.h"
#undef TEST_IMPL_H

int main(int argc, char *argv[]) {
  char *backend;
  int device, platform;
  parse_input(argc, argv, &backend, &device, &platform);

  nomp_api_233_int(backend, device, platform);
  nomp_api_233_long(backend, device, platform);
  nomp_api_233_unsigned(backend, device, platform);
  nomp_api_233_unsigned_long(backend, device, platform);
  nomp_api_233_float(backend, device, platform);
  nomp_api_233_double(backend, device, platform);

  return 0;
}
