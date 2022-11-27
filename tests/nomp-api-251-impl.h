#include "nomp-test.h"

#define nomp_api_251_aux TOKEN_PASTE(nomp_api_251_aux, TEST_SUFFIX)
int nomp_api_251_aux(TEST_TYPE *a, TEST_TYPE *b, int E, int N) {
  const char *knl_fmt =
      "void foo(%s *a, %s *b, int E, int N) {                 \n"
      "  for (int e = 0; e < E; e++)                          \n"
      "    for (int i = 0; i < N; i++)                        \n"
      "      a[e * N + i] = a[e * N + i] + b[e * N + i];      \n"
      "}                                                      \n";

  const char *test_type = TOSTRING(TEST_TYPE);
  size_t len = strlen(knl_fmt) + 2 * strlen(test_type) + 1;
  char *knl = tcalloc(char, len);
  snprintf(knl, len, knl_fmt, test_type, test_type);

  static int id = -1;
  const char *clauses[7] = {"annotate",     "dof_loop", "i", "annotate",
                            "element_loop", "e",        0};
  int err = nomp_jit(&id, knl, clauses);
  nomp_chk(err);

  err = nomp_run(id, 4, "a", NOMP_PTR, sizeof(TEST_TYPE), a, "b", NOMP_PTR,
                 sizeof(TEST_TYPE), b, "E", NOMP_INTEGER, sizeof(int), &E, "N",
                 NOMP_INTEGER, sizeof(int), &N);
  nomp_chk(err);
  tfree(knl);

  return 0;
}

#define nomp_api_251 TOKEN_PASTE(nomp_api_251, TEST_SUFFIX)
int nomp_api_251(int argc, const char **argv) {
  int err = nomp_init(argc, (const char **)argv);
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
  nomp_api_251_aux(a, b, E, N);

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
#undef nomp_api_251

#undef nomp_api_251_aux
