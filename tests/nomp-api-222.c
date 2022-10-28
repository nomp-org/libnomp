#include "nomp.h"

#define TEST_TYPE int
#define TEST_SUFFIX _int
#include "nomp-api-222-impl.h"
#undef TEST_TYPE
#undef TEST_SUFFIX

#define TEST_TYPE long
#define TEST_SUFFIX _long
#include "nomp-api-222-impl.h"
#undef TEST_TYPE
#undef TEST_SUFFIX

#define TEST_TYPE unsigned
#define TEST_SUFFIX _unsigned
#include "nomp-api-222-impl.h"
#undef TEST_TYPE
#undef TEST_SUFFIX

#define TEST_TYPE unsigned long
#define TEST_SUFFIX _unsigned_long
#include "nomp-api-222-impl.h"
#undef TEST_TYPE
#undef TEST_SUFFIX

#define TEST_TOL 1e-12
#define TEST_TYPE double
#define TEST_SUFFIX _double
#include "nomp-api-222-impl.h"
#undef TEST_TYPE
#undef TEST_SUFFIX
#undef TEST_TOL

#define TEST_TOL 1e-8
#define TEST_TYPE float
#define TEST_SUFFIX _float
#include "nomp-api-222-impl.h"
#undef TEST_TYPE
#undef TEST_SUFFIX
#undef TEST_TOL

int main(int argc, char *argv[]) {
  char *backend = argc > 1 ? argv[1] : "opencl";
  int device = argc > 2 ? atoi(argv[2]) : 0;
  int platform = argc > 3 ? atoi(argv[3]) : 0;

  int err = nomp_init(backend, platform, device);
  nomp_chk(err);

  nomp_api_222_int();
  nomp_api_222_long();
  nomp_api_222_unsigned();
  nomp_api_222_unsigned_long();
  nomp_api_222_float();
  nomp_api_222_double();

  err = nomp_finalize();
  nomp_chk(err);

  return 0;
}
