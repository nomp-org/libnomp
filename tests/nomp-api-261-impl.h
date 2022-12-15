#include "nomp-test.h"

#define nomp_api_261_aux TOKEN_PASTE(nomp_api_261_aux, TEST_SUFFIX)
int nomp_api_261_aux(TEST_TYPE *a, TEST_TYPE *b, int E, int N) {
  const char *KNL_FMT =
      "void foo(%s *a, %s *b, int E, int N) {                 \n"
      "  for (int e = 0; e < E; e++)                          \n"
      "    for (int i = 0; i < N; i++)                        \n"
      "      a[e * N + i] = a[e * N + i] + b[e * N + i];      \n"
      "}                                                      \n";

  const char *TYPE_STR = TOSTRING(TEST_TYPE);
  size_t len = strlen(KNL_FMT) + 2 * strlen(TYPE_STR) + 1;
  char *knl = tcalloc(char, len);
  snprintf(knl, len, KNL_FMT, TYPE_STR, TYPE_STR);

  static int id = -1;
  const char *clauses[7] = {"annotate",     "dof_loop", "i", "annotate",
                            "element_loop", "e",        0};
  int err = nomp_jit(&id, knl, clauses, 4, "a", NOMP_PTR, sizeof(TEST_TYPE),
                     "b", NOMP_PTR, sizeof(TEST_TYPE), "E", NOMP_INT,
                     sizeof(int), "N", NOMP_INT, sizeof(int));
  tfree(knl);
  nomp_chk(err);

  err = nomp_run(id, a, b, &E, &N);
  nomp_chk(err);

  return 0;
}

#define nomp_api_261 TOKEN_PASTE(nomp_api_261, TEST_SUFFIX)
int nomp_api_261(int argc, const char **argv) {
  int err = nomp_init(argc, argv);
  nomp_chk(err);

  TEST_TYPE a[128], b[128];
  const int n = 128;
  for (unsigned i = 0; i < n; i++)
    a[i] = 2 * n - i, b[i] = i;

  err = nomp_update(a, 0, n, sizeof(TEST_TYPE), NOMP_TO);
  nomp_chk(err);
  err = nomp_update(b, 0, n, sizeof(TEST_TYPE), NOMP_TO);
  nomp_chk(err);

  const int E = 4;
  const int N = 32;
  nomp_api_261_aux(a, b, E, N);

  err = nomp_update(a, 0, n, sizeof(TEST_TYPE), NOMP_FROM);
  nomp_chk(err);

#if defined(TEST_TOL)
  for (unsigned i = 0; i < n; i++)
    nomp_assert(fabs(a[i] - 2 * n) < TEST_TOL);
#else
  for (unsigned i = 0; i < n; i++)
    nomp_assert(a[i] == 2 * n);
#endif

  err = nomp_update(a, 0, n, sizeof(TEST_TYPE), NOMP_FREE);
  nomp_chk(err);
  err = nomp_update(b, 0, n, sizeof(TEST_TYPE), NOMP_FREE);
  nomp_chk(err);

  err = nomp_finalize();
  nomp_chk(err);

  return 0;
}
#undef nomp_api_261

#undef nomp_api_261_aux
