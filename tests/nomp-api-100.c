#include "nomp.h"
#include <stdlib.h>

#define TEST_IMPL_H "nomp-api-100-impl.h"
#include "nomp-generate-tests.h"
#undef TEST_IMPL_H

int main(int argc, char *argv[]) {
  char *backend = argc > 1 ? argv[1] : "opencl";
  int device = argc > 2 ? atoi(argv[2]) : 0;
  int platform = argc > 3 ? atoi(argv[3]) : 0;

  int err = nomp_init(backend, device, platform);
  nomp_chk(err);

  nomp_api_100_int(0, 10);
  nomp_api_100_long(0, 10);
  nomp_api_100_unsigned(0, 10);
  nomp_api_100_unsigned_long(0, 10);
  nomp_api_100_double(0, 10);
  nomp_api_100_float(0, 10);

  nomp_api_100_int(5, 10);
  nomp_api_100_long(5, 10);
  nomp_api_100_unsigned(5, 10);
  nomp_api_100_unsigned_long(5, 10);
  nomp_api_100_double(5, 10);
  nomp_api_100_float(5, 10);

  nomp_api_100_int(2, 8);
  nomp_api_100_long(2, 8);
  nomp_api_100_unsigned(2, 8);
  nomp_api_100_unsigned_long(2, 8);
  nomp_api_100_double(2, 8);
  nomp_api_100_float(2, 8);

  err = nomp_finalize();
  nomp_chk(err);

  return 0;
}
