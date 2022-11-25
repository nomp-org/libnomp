#include "nomp.h"

#define TEST_IMPL_H "nomp-api-223-impl.h"
#include "nomp-generate-tests.h"
#undef TEST_IMPL_H

int main(int argc, char *argv[]) {
  char *backend = argc > 1 ? argv[1] : "opencl";
  int device = argc > 2 ? atoi(argv[2]) : 0;
  int platform = argc > 3 ? atoi(argv[3]) : 0;

  int err = nomp_init(backend, platform, device);
  nomp_chk(err);

  nomp_api_223_add_int();
  nomp_api_223_add_long();
  nomp_api_223_add_unsigned();
  nomp_api_223_add_unsigned_long();
  nomp_api_223_add_float();
  nomp_api_223_add_double();

  nomp_api_223_sub_int();
  nomp_api_223_sub_long();
  nomp_api_223_sub_unsigned();
  nomp_api_223_sub_unsigned_long();
  nomp_api_223_sub_float();
  nomp_api_223_sub_double();

  nomp_api_223_mul_int();
  nomp_api_223_mul_long();
  nomp_api_223_mul_unsigned();
  nomp_api_223_mul_unsigned_long();
  nomp_api_223_mul_float();
  nomp_api_223_mul_double();

  err = nomp_finalize();
  nomp_chk(err);

  return 0;
}
