#include "nomp.h"

#define TEST_IMPL_H "nomp-api-220-impl.h"
#include "nomp-generate-tests.h"
#undef TEST_IMPL_H

int main(int argc, char *argv[]) {
  char *backend;
  int device, platform;
  parse_input(argc, argv, &backend, &device, &platform);

  int err = nomp_init(backend, platform, device);
  nomp_chk(err);

  nomp_api_220_int(10);
  nomp_api_220_long(10);
  nomp_api_220_unsigned(10);
  nomp_api_220_unsigned_long(10);
  nomp_api_220_float(10);
  nomp_api_220_double(10);

  nomp_api_220_int(20);
  nomp_api_220_long(20);
  nomp_api_220_unsigned(20);
  nomp_api_220_unsigned_long(20);
  nomp_api_220_float(20);
  nomp_api_220_double(20);

  err = nomp_finalize();
  nomp_chk(err);

  return 0;
}
