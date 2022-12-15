#define TEST_IMPL_H "nomp-api-250-impl.h"
#include "nomp-generate-tests.h"
#undef TEST_IMPL_H

int main(int argc, const char *argv[]) {
  const char *args[] = {"-b", "opencl", "-d",  "0",   "-p",
                        "0",  "-as",    "sem", "-af", "annotate"};
  argc = 10;
  TEST_BUILTIN_TYPES(250, argc, args)

  return 0;
}
