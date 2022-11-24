#include "nomp.h"

#define TEST_IMPL_H "nomp-api-251-impl.h"
#include "nomp-generate-tests.h"
#undef TEST_IMPL_H

int main(int argc,const char *argv[]) {

  TEST_BUILTIN_TYPES(251, argc, argv)

  return 0;
}
