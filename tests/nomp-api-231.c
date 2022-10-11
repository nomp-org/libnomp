#include "nomp.h"
#include <stdlib.h>

#define TEST_TYPE int
#define TEST_SUFFIX _int
#include "nomp-api-231-impl.h"
#undef TEST_TYPE
#undef TEST_SUFFIX

#define TEST_TYPE long
#define TEST_SUFFIX _long
#include "nomp-api-231-impl.h"
#undef TEST_TYPE
#undef TEST_SUFFIX

#define TEST_TYPE unsigned
#define TEST_SUFFIX _unsigned
#include "nomp-api-231-impl.h"
#undef TEST_TYPE
#undef TEST_SUFFIX

#define TEST_TYPE unsigned long
#define TEST_SUFFIX _unsigned_long
#include "nomp-api-231-impl.h"
#undef TEST_TYPE
#undef TEST_SUFFIX

#define TEST_TOL 1e-12
#define TEST_TYPE double
#define TEST_SUFFIX _double
#include "nomp-api-231-impl.h"
#undef TEST_TYPE
#undef TEST_SUFFIX
#undef TEST_TOL

#define TEST_TOL 1e-8
#define TEST_TYPE float
#define TEST_SUFFIX _float
#include "nomp-api-231-impl.h"
#undef TEST_TYPE
#undef TEST_SUFFIX
#undef TEST_TOL

int main(int argc, char *argv[]) {
  char *backend = argc > 1 ? argv[1] : "opencl";
  int device = argc > 2 ? atoi(argv[2]) : 0;
  int platform = argc > 3 ? atoi(argv[3]) : 0;

  nomp_api_231_int(backend, device, platform);
  nomp_api_231_long(backend, device, platform);
  nomp_api_231_unsigned(backend, device, platform);
  nomp_api_231_unsigned_long(backend, device, platform);
  nomp_api_231_float(backend, device, platform);
  nomp_api_231_double(backend, device, platform);

  return 0;
}
