#define TEST_IMPL_H "nomp-api-230-impl.h"
#include "nomp-generate-tests.h"
#undef TEST_IMPL_H

int main(int argc,const char *argv[]) {

  TEST_BUILTIN_TYPES(230, argc, argv)

  return 0;
}
