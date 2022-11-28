#include "nomp.h"

#define TEST_IMPL_H "nomp-api-222-impl.h"
#include "nomp-generate-tests.h"
#undef TEST_IMPL_H

int main(int argc,const char *argv[]) {

  int err = nomp_init(argc, argv);
  nomp_chk(err);

  TEST_BUILTIN_TYPES(222, 10)
  TEST_BUILTIN_TYPES(222, 20)

  err = nomp_finalize();
  nomp_chk(err);

  return 0;
}
