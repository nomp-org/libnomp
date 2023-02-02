#include "nomp-aux.h"
#include "nomp-jit.h"
#include "nomp-test.h"

static int test_jit_compile_and_free() {
  const char *source = "int foo(int a, int b) {                             \n"
                       "  return a + b;                                     \n"
                       "}                                                   \n"
                       "                                                    \n"
                       "void foo_wrapper(void **p) {                        \n"
                       "  int a = *((int *)p[0]);                           \n"
                       "  int b = *((int *)p[1]);                           \n"
                       "  *((int *)p[2]) = foo(a, b);                       \n"
                       "}                                                   \n";

  char cwd[BUFSIZ];
  if (getcwd(cwd, BUFSIZ) == NULL) {
    perror("getcwd() error");
    return 1;
  }

  const char *cc = "/usr/bin/cc";
  const char *cflags = "-shared";
  const char *entry = "foo_wrapper";
  const char *wkdir = strcatn(3, BUFSIZ, cwd, "/", ".nomp_jit_cache");
  int id = -1;

  int err = jit_compile(&id, source, cc, cflags, entry, wkdir);
  nomp_test_chk(err);
  nomp_test_assert(id == 0);

  err = jit_free(&id);
  nomp_test_chk(err);
  nomp_test_assert(id == -1);

  return 0;
}

int main(int argc, const char *argv[]) {
  int err = 0;
  err |= SUBTEST(test_jit_compile_and_free);

  return err;
}
