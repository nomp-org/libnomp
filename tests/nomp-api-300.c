#include "nomp.h"
#include <stdlib.h>

#define TEST_IMPL_H "nomp-api-300-impl.h"
#include "nomp-generate-tests.h"
#undef TEST_IMPL_H

int main(int argc, char *argv[]) {
  char *backend = argc > 1 ? argv[1] : "opencl";
  int device = argc > 2 ? atoi(argv[2]) : 0;
  int platform = argc > 3 ? atoi(argv[3]) : 0;

  int err = nomp_init(backend, platform, device);
  nomp_chk(err);

  nomp_api_300_int(3, 2);
  nomp_api_300_long(3, 2);
  nomp_api_300_unsigned(3, 2);
  nomp_api_300_unsigned_long(3, 2);
  nomp_api_300_float(3, 2);
  nomp_api_300_double(3, 2);

  nomp_api_300_int(10, 10);
  nomp_api_300_long(10, 10);
  nomp_api_300_unsigned(10, 10);
  nomp_api_300_unsigned_long(10, 10);
  nomp_api_300_float(10, 10);
  nomp_api_300_double(10, 10);

  err = nomp_finalize();
  nomp_chk(err);

  return 0;
}