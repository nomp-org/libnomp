#include "nomp-test.h"

#define nomp_api_233_aux TOKEN_PASTE(nomp_api_233_aux, TEST_SUFFIX)
int nomp_api_233_aux(TEST_TYPE *a, TEST_TYPE *b, TEST_TYPE *c, int N) {
  const char *knl_fmt =
      "void foo(%s *a, %s *b, %s *c, int N) {                 \n"
      "  for (int i = 0; i < N; i++)                          \n"
      "    a[i] = a[i] * b[i] + c[i];                         \n"
      "}                                                      \n";

  size_t len = strlen(knl_fmt) + 3 * strlen(TOSTRING(TEST_TYPE)) + 1;
  char *knl = (char *)calloc(len, sizeof(char));
  snprintf(knl, len, knl_fmt, TOSTRING(TEST_TYPE), TOSTRING(TEST_TYPE),
           TOSTRING(TEST_TYPE));

  static int id = -1;
  const char *annotations[1] = {0},
             *clauses[3] = {"transform", "nomp-api-200:transform", 0};
  int err = nomp_jit(&id, knl, annotations, clauses);
  nomp_chk(err);

  err = nomp_run(id, 4, "a", NOMP_PTR, sizeof(TEST_TYPE), a, "b", NOMP_PTR,
                 sizeof(TEST_TYPE), b, "c", NOMP_PTR, sizeof(TEST_TYPE), c, "N",
                 NOMP_INTEGER, sizeof(int), &N);
  nomp_chk(err);

  free(knl);
  return 0;
}

#define nomp_api_233 TOKEN_PASTE(nomp_api_233, TEST_SUFFIX)
int nomp_api_233(const char *backend, int device, int platform) {
  int err = nomp_init(backend, platform, device);
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
