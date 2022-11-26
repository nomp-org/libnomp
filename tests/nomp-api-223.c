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

  nomp_api_223_add_int(10);
  nomp_api_223_add_long(10);
  nomp_api_223_add_unsigned(10);
  nomp_api_223_add_unsigned_long(10);
  nomp_api_223_add_float(10);
  nomp_api_223_add_double(10);

  nomp_api_223_add_int(20);
  nomp_api_223_add_long(20);
  nomp_api_223_add_unsigned(20);
  nomp_api_223_add_unsigned_long(20);
  nomp_api_223_add_float(20);
  nomp_api_223_add_double(20);

  nomp_api_223_sub_int(10);
  nomp_api_223_sub_long(10);
  nomp_api_223_sub_unsigned(10);
  nomp_api_223_sub_unsigned_long(10);
  nomp_api_223_sub_float(10);
  nomp_api_223_sub_double(10);

  nomp_api_223_sub_int(20);
  nomp_api_223_sub_long(20);
  nomp_api_223_sub_unsigned(20);
  nomp_api_223_sub_unsigned_long(20);
  nomp_api_223_sub_float(20);
  nomp_api_223_sub_double(20);

  nomp_api_223_mul_int(10);
  nomp_api_223_mul_long(10);
  nomp_api_223_mul_unsigned(10);
  nomp_api_223_mul_unsigned_long(10);
  nomp_api_223_mul_float(10);
  nomp_api_223_mul_double(10);

  nomp_api_223_mul_int(20);
  nomp_api_223_mul_long(20);
  nomp_api_223_mul_unsigned(20);
  nomp_api_223_mul_unsigned_long(20);
  nomp_api_223_mul_float(20);
  nomp_api_223_mul_double(20);

  err = nomp_finalize();
  nomp_chk(err);

  return 0;
}
