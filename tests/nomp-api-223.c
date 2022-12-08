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

  TEST_BUILTIN_TYPES(223_add, 10)
  TEST_BUILTIN_TYPES(223_add, 20)
  TEST_BUILTIN_TYPES(223_sub, 10)
  TEST_BUILTIN_TYPES(223_sub, 20)
  TEST_BUILTIN_TYPES(223_mul, 20)
  TEST_BUILTIN_TYPES(223_mul, 20)

  err = nomp_finalize();
  nomp_chk(err);

  return 0;
}
