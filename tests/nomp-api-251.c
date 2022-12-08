#include "nomp.h"

#define TEST_IMPL_H "nomp-api-251-impl.h"
#include "nomp-generate-tests.h"
#undef TEST_IMPL_H

int main(int argc,const char *argv[]) {
  const char *args[] = {"-b", "opencl", "-d",  "0",   "-p",
                        "0",  "-as",    "sem", "-af", "annotate"};
  argc = 11;
  TEST_BUILTIN_TYPES(251, argc, args)

  return 0;
}
