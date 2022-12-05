#include "nomp-test.h"

#define nomp_api_210 TOKEN_PASTE(nomp_api_210, TEST_SUFFIX)
int nomp_api_210() {
  int N = 20;
  TEST_TYPE a[20] = {0}, b[20] = {1, 2, 3, 4, 5};

  int err = nomp_update(a, 0, 20, sizeof(TEST_TYPE), NOMP_TO);
  nomp_chk(err);
  err = nomp_update(b, 0, 20, sizeof(TEST_TYPE), NOMP_TO);
  nomp_chk(err);

  const char *KNL_FMT =
      "void foo(%s *a, %s *b, int N) {                        \n"
      "  for (int i = 0; i < N; i++)                          \n"
      "    a[i] = 2 * b[i] + 1;                               \n"
      "}                                                      \n";

  const char *TYPE_STR = TOSTRING(TEST_TYPE);
  size_t len = strlen(KNL_FMT) + 2 * strlen(TYPE_STR) + 1;
  char *knl = tcalloc(char, len);
  snprintf(knl, len, KNL_FMT, TYPE_STR, TYPE_STR);

  static int id = -1;
  const char *clauses[4] = {"transform", "nomp-api-200", "foo", 0};
  err = nomp_jit(&id, knl, clauses, 3, "a", NOMP_PTR, sizeof(TEST_TYPE), "b",
                 NOMP_PTR, sizeof(TEST_TYPE), "N", NOMP_INT, sizeof(int));
  tfree(knl);
  nomp_chk(err);

  err = nomp_run(id, a, b, &N);
  nomp_chk(err);

  err = nomp_update(a, 0, 20, sizeof(TEST_TYPE), NOMP_FROM);
  nomp_chk(err);

#if defined(TEST_TOL)
  for (int i = 0; i < N; i++)
    nomp_assert(fabs(a[i] - 2 * b[i] - 1) < TEST_TOL);
#else
  for (int i = 0; i < N; i++)
    nomp_assert(a[i] == 2 * b[i] + 1);
#endif

  err = nomp_update(a, 0, 20, sizeof(TEST_TYPE), NOMP_FREE);
  nomp_chk(err);
  err = nomp_update(b, 0, 20, sizeof(TEST_TYPE), NOMP_FREE);
  nomp_chk(err);

  return 0;
}
#undef nomp_api_210
