#include "nomp.h"

#define TEST_IMPL_H "nomp-api-220-impl.h"
#include "nomp-generate-tests.h"
#undef TEST_IMPL_H

int main(int argc, char *argv[]) {
  char *backend = argc > 1 ? argv[1] : "opencl";
  int device = argc > 2 ? atoi(argv[2]) : 0;
  int platform = argc > 3 ? atoi(argv[3]) : 0;

  int err = nomp_init(backend, platform, device);
  nomp_chk(err);

  nomp_api_220_int();
  nomp_api_220_long();
  nomp_api_220_unsigned();
  nomp_api_220_unsigned_long();
  nomp_api_220_float();
  nomp_api_220_double();

  nomp_api_220_no_free_int();
  nomp_api_220_no_free_long();
  nomp_api_220_no_free_unsigned();
  nomp_api_220_no_free_unsigned_long();
  nomp_api_220_no_free_float();
  nomp_api_220_no_free_double();

  err = nomp_finalize();
  nomp_chk(err);

  return 0;
}
