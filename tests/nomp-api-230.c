#include "nomp.h"

#define TEST_IMPL_H "nomp-api-230-impl.h"
#include "nomp-generate-tests.h"
#undef TEST_IMPL_H

int main(int argc, char *argv[]) {
  char *backend = argc > 1 ? argv[1] : "opencl";
  int device = argc > 2 ? atoi(argv[2]) : 0;
  int platform = argc > 3 ? atoi(argv[3]) : 0;

  nomp_api_230_int(backend, device, platform);
  nomp_api_230_long(backend, device, platform);
  nomp_api_230_unsigned(backend, device, platform);
  nomp_api_230_unsigned_long(backend, device, platform);
  nomp_api_230_float(backend, device, platform);
  nomp_api_230_double(backend, device, platform);

  return 0;
}
