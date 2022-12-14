#include "nomp.h"

#define TEST_IMPL_H "nomp-api-400-impl.h"
#include "nomp-generate-tests.h"
#undef TEST_IMPL_H

int main(int argc, const char *argv[]) {
  nomp_api_400_int(argc, argv);

  return 0;
}
