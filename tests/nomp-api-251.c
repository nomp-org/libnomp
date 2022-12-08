#include "nomp.h"

#define TEST_IMPL_H "nomp-api-251-impl.h"
#include "nomp-generate-tests.h"
#undef TEST_IMPL_H

int main(int argc, char *argv[]) {
  char *backend = argc > 1 ? argv[1] : "opencl";
  int device = argc > 2 ? atoi(argv[2]) : 0;
  int platform = argc > 3 ? atoi(argv[3]) : 0;

  TEST_BUILTIN_TYPES(251, backend, device, platform)

  return 0;
}
