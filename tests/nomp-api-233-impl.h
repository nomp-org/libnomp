#include "nomp-test.h"
#include "nomp.h"

#define nomp_api_233_aux TOKEN_PASTE(nomp_api_233_aux, TEST_SUFFIX)
int nomp_api_233_aux(TEST_TYPE *a, TEST_TYPE *b, TEST_TYPE *c, int N) {
  const char *KNL_FMT =
      "void foo(%s *a, %s *b, %s *c, int N) {                 \n"
      "  for (int i = 0; i < N; i++)                          \n"
      "    a[i] = a[i] * b[i] + c[i];                         \n"
      "}                                                      \n";

  const char *TYPE_STR = TOSTRING(TEST_TYPE);
  size_t len = strlen(KNL_FMT) + 3 * strlen(TYPE_STR) + 1;
  char *knl = tcalloc(char, len);
  snprintf(knl, len, KNL_FMT, TYPE_STR, TYPE_STR, TYPE_STR);

  static int id = -1;
  const char *clauses[4] = {"transform", "nomp-api-200", "foo", 0};
  int err = nomp_jit(&id, knl, clauses, 4, "a", NOMP_PTR, sizeof(TEST_TYPE),
                     "b", NOMP_PTR, sizeof(TEST_TYPE), "c", NOMP_PTR,
                     sizeof(TEST_TYPE), "N", NOMP_INT, sizeof(int));
  tfree(knl);
  nomp_chk(err);

  err = nomp_run(id, a, b, c, &N);
  nomp_chk(err);

  return 0;
}

#define nomp_api_233 TOKEN_PASTE(nomp_api_233, TEST_SUFFIX)
int nomp_api_233(int argc, const char *argv[]) {
  int err = nomp_init(argc, argv);
  nomp_chk(err);

  int n = 10;
  TEST_TYPE a[10], b[10], c[10];
  for (unsigned i = 0; i < n; i++)
    a[i] = n - i, b[i] = i, c[i] = 5;

  err = nomp_update(a, 0, n, sizeof(TEST_TYPE), NOMP_TO);
  nomp_chk(err);
  err = nomp_update(b, 0, n, sizeof(TEST_TYPE), NOMP_TO);
  nomp_chk(err);
  err = nomp_update(c, 0, n, sizeof(TEST_TYPE), NOMP_TO);
  nomp_chk(err);

  nomp_api_233_aux(a, b, c, n);

  err = nomp_update(a, 0, n, sizeof(TEST_TYPE), NOMP_FROM);
  nomp_chk(err);

#if defined(TEST_TOL)
  for (unsigned i = 0; i < n; i++)
    nomp_assert(fabs(a[i] - (n - i) * i - 5) < TEST_TOL);
#else
  for (unsigned i = 0; i < n; i++)
    nomp_assert(a[i] == (n - i) * i + 5);
#endif

  err = nomp_update(a, 0, n, sizeof(TEST_TYPE), NOMP_FREE);
  nomp_chk(err);
  err = nomp_update(b, 0, n, sizeof(TEST_TYPE), NOMP_FREE);
  nomp_chk(err);
  err = nomp_update(c, 0, n, sizeof(TEST_TYPE), NOMP_FREE);
  nomp_chk(err);

  err = nomp_finalize();
  nomp_chk(err);

  return 0;
}
#undef nomp_api_233

#undef nomp_api_233_aux
