#include "nomp-test.h"

#define nomp_api_251_aux TOKEN_PASTE(nomp_api_251_aux, TEST_SUFFIX)
int nomp_api_251_aux(TEST_TYPE *a, TEST_TYPE *b, int E, int N) {
  const char *knl_fmt =
      "void foo(%s *a, %s *b, int E, int N) {                 \n"
      "  for (int e = 0; e < E; e++)                          \n"
      "    for (int i = 0; i < N; i++)                        \n"
      "      a[e * N + i] = a[e * N + i] + b[e * N + i];      \n"
      "}                                                      \n";
  const char *clauses[7] = {"annotate",     "dof_loop", "i", "annotate",
                            "element_loop", "e",        0};

  char *knl = create_knl(knl_fmt, 2, TOSTRING(TEST_TYPE), TOSTRING(TEST_TYPE));

  return run_kernel(knl, clauses, 4, "a", NOMP_PTR, sizeof(TEST_TYPE), a, "b", NOMP_PTR,
                 sizeof(TEST_TYPE), b, "E", NOMP_INTEGER, sizeof(int), &E, "N",
                 NOMP_INTEGER, sizeof(int), &N);
}

#define nomp_api_251 TOKEN_PASTE(nomp_api_251, TEST_SUFFIX)
int nomp_api_251(int argc, const char **argv) {
  int err = nomp_init(argc, argv);
  nomp_test_chk(err);

  TEST_TYPE a[128], b[128];
  const int n = 128;
  for (unsigned i = 0; i < n; i++)
    a[i] = 2 * n - i, b[i] = i;

  err = nomp_update(a, 0, n, sizeof(TEST_TYPE), NOMP_TO);
  nomp_test_chk(err);
  err = nomp_update(b, 0, n, sizeof(TEST_TYPE), NOMP_TO);
  nomp_test_chk(err);

  const int E = 4;
  const int N = 32;
  nomp_api_251_aux(a, b, E, N);

  err = nomp_update(a, 0, n, sizeof(TEST_TYPE), NOMP_FROM);
  nomp_test_chk(err);

#if defined(TEST_TOL)
  for (unsigned i = 0; i < n; i++)
    nomp_test_assert(fabs(a[i] - 2 * n) < TEST_TOL);
#else
  for (unsigned i = 0; i < n; i++)
    nomp_test_assert(a[i] == 2 * n);
#endif

  err = nomp_update(a, 0, n, sizeof(TEST_TYPE), NOMP_FREE);
  nomp_test_chk(err);
  err = nomp_update(b, 0, n, sizeof(TEST_TYPE), NOMP_FREE);
  nomp_test_chk(err);

  err = nomp_finalize();
  nomp_test_chk(err);

  return 0;
}
#undef nomp_api_251

#undef nomp_api_251_aux
