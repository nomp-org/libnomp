#include "nomp.h"
#include <stdlib.h>

#define TEST_IMPL_H "nomp-api-240-impl.h"
#include "nomp-generate-tests.h"
#undef TEST_IMPL_H

int main(int argc, char *argv[]) {
  char *backend = argc > 1 ? argv[1] : "opencl";
  int device = argc > 2 ? atoi(argv[2]) : 0;
  int platform = argc > 3 ? atoi(argv[3]) : 0;

  int err = nomp_init(backend, device, platform);
  nomp_chk(err);

  nomp_api_240_int(10);
  nomp_api_240_long(10);
  nomp_api_240_unsigned(10);
  nomp_api_240_unsigned_long(10);
  nomp_api_240_float(10);
  nomp_api_240_double(10);

  nomp_api_240_int(20);
  nomp_api_240_long(20);
  nomp_api_240_unsigned(20);
  nomp_api_240_unsigned_long(20);
  nomp_api_240_float(20);
  nomp_api_240_double(20);

  err = nomp_finalize();
  nomp_chk(err);

  return 0;
}
