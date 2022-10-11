#include "nomp.h"
#include <stdlib.h>

#define TEST_TYPE int
#define TEST_SUFFIX _int
#include "nomp-api-100-impl.h"
#undef TEST_TYPE
#undef TEST_SUFFIX

#define TEST_TYPE long
#define TEST_SUFFIX _long
#include "nomp-api-100-impl.h"
#undef TEST_TYPE
#undef TEST_SUFFIX

#define TEST_TYPE unsigned
#define TEST_SUFFIX _unsigned
#include "nomp-api-100-impl.h"
#undef TEST_TYPE
#undef TEST_SUFFIX

#define TEST_TYPE unsigned long
#define TEST_SUFFIX _unsigned_long
#include "nomp-api-100-impl.h"
#undef TEST_TYPE
#undef TEST_SUFFIX

#define TEST_TOL 1e-12
#define TEST_TYPE double
#define TEST_SUFFIX _double
#include "nomp-api-100-impl.h"
#undef TEST_TYPE
#undef TEST_SUFFIX
#undef TEST_TOL

#define TEST_TOL 1e-8
#define TEST_TYPE float
#define TEST_SUFFIX _float
#include "nomp-api-100-impl.h"
#undef TEST_TYPE
#undef TEST_SUFFIX
#undef TEST_TOL

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
