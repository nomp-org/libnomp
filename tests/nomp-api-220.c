#include "nomp.h"

#define TEST_IMPL_H "nomp-api-220-impl.h"
#include "nomp-generate-tests.h"
#undef TEST_IMPL_H

int main(int argc, char *argv[]) {
  char *backend;
  int device, platform;
  parse_input(argc, argv, &backend, &device, &platform);

  int err = nomp_init(&argc, &argv);
  nomp_chk(err);

  TEST_BUILTIN_TYPES(220, 10)
  TEST_BUILTIN_TYPES(220, 20)

  err = nomp_finalize();
  nomp_chk(err);

  return 0;
}
