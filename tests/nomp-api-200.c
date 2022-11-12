#include "nomp.h"

#define TEST_IMPL_H "nomp-api-200-impl.h"
#include "nomp-generate-tests.h"
#undef TEST_IMPL_H

int main(int argc, char *argv[]) {
  char *backend;
  int device, platform;
  parse_input(argc, argv, &backend, &device, &platform);

  int err = nomp_init(backend, platform, device);
  nomp_chk(err);

  nomp_api_200_int();
  nomp_api_200_long();
  nomp_api_200_unsigned();
  nomp_api_200_unsigned_long();
  nomp_api_200_double();
  nomp_api_200_float();

  err = nomp_finalize();
  nomp_chk(err);

  return 0;
}
