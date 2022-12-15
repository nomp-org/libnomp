#include "nomp.h"

#define TEST_IMPL_H "nomp-api-260-impl.h"
#include "nomp-generate-tests.h"
#undef TEST_IMPL_H

int main(int argc, const char *argv[]) {
  const char *args[] = {"-b", "opencl", "-d",  "0",   "-p",
                        "0",  "-as",    "sem", "-af", "annotate"};
  argc = 10;
  TEST_BUILTIN_TYPES(260, argc, args)

  return 0;
}
