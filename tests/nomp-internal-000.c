#include "nomp-aux.h"
#include "nomp-jit.h"
#include "nomp-test.h"

static const char *add_src =
    "int add(int a, int b) {                             \n"
    "  return a + b;                                     \n"
    "}                                                   \n"
    "                                                    \n"
    "void add_wrapper(void **p) {                        \n"
    "  int a = *((int *)p[0]);                           \n"
    "  int b = *((int *)p[1]);                           \n"
    "  *((int *)p[2]) = add(a, b);                       \n"
    "}                                                   \n";

static int test_jit_compile_and_free(const char *cwd, const char *wkdir) {
  const char *cc = "/usr/bin/cc", *cflags = "-shared", *entry = "add_wrapper";
  const char *srcf = "source.c", *libf = "mylib.so";
  int id = -1;
  int err = jit_compile(&id, add_src, cc, cflags, entry, wkdir, srcf, libf,
                        NOMP_WRITE, NOMP_DO_NOT_OVERWRITE);
  nomp_test_chk(err);
  nomp_test_assert(id == 0);

  err = jit_free(&id);
  nomp_test_chk(err);
  nomp_test_assert(id == -1);

  return 0;
}

static int test_jit_run(const char *cwd, const char *wkdir) {
  const char *cc = "/usr/bin/cc", *cflags = "-shared", *entry = "add_wrapper";
  const char *srcf = "source.c", *libf = "mylib.so";
  int id = -1;
  int err = jit_compile(&id, add_src, cc, cflags, entry, wkdir, srcf, libf,
                        NOMP_WRITE, NOMP_DO_NOT_OVERWRITE);
  nomp_test_chk(err);

  int a = 3, b = 7, c = -1;
  void *p[3] = {(void *)&a, (void *)&b, (void *)&c};
  err = jit_run(id, p);
  nomp_test_chk(err);
  nomp_test_assert(c == 10);

  err = jit_free(&id);
  nomp_test_chk(err);

  return 0;
}

int main(int argc, const char *argv[]) {
  char cwd[BUFSIZ];
  if (getcwd(cwd, BUFSIZ) == NULL) {
    perror("getcwd() error");
    return 1;
  }

  char *wkdir = nomp_str_cat(3, BUFSIZ, cwd, "/", ".nomp_jit_cache");

  int err = 0;
  err |= SUBTEST(test_jit_compile_and_free, cwd, wkdir);
  err |= SUBTEST(test_jit_run, cwd, wkdir);

  nomp_free(wkdir);

  return err;
}
