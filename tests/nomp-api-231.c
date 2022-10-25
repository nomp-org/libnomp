#include "nomp.h"
#include <stdlib.h>

#define TEST_IMPL_H "nomp-api-231-impl.h"
#include "nomp-generate-tests.h"
#undef TEST_IMPL_H

int main(int argc, char *argv[]) {
  char *backend = argc > 1 ? argv[1] : "opencl";
  int device = argc > 2 ? atoi(argv[2]) : 0;
  int platform = argc > 3 ? atoi(argv[3]) : 0;

  nomp_api_231_int(backend, device, platform);
  nomp_api_231_long(backend, device, platform);
  nomp_api_231_unsigned(backend, device, platform);
  nomp_api_231_unsigned_long(backend, device, platform);
  nomp_api_231_float(backend, device, platform);
  nomp_api_231_double(backend, device, platform);

  return 0;
}