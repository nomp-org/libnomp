#include "nomp-test.h"

#if defined(TEST_TOL)
#include "math.h"
#endif

#define nomp_api_210 TOKEN_PASTE(nomp_api_210, TEST_SUFFIX)
int nomp_api_210() {
  TEST_TYPE a[20] = {0}, b[20] = {1, 2, 3, 4, 5};
  int N = 20;

  int err = nomp_update(a, 0, 20, sizeof(TEST_TYPE), NOMP_TO);
  nomp_chk(err);
  err = nomp_update(b, 0, 20, sizeof(TEST_TYPE), NOMP_TO);
  nomp_chk(err);

  const char *knl_fmt =
      "void foo(%s *a, %s *b, int N) {                        \n"
      "  for (int i = 0; i < N; i++)                          \n"
      "    a[i] = 2 * b[i] + 1;                               \n"
      "}                                                      \n";

  size_t len = strlen(knl_fmt) + 2 * strlen(TOSTRING(TEST_TYPE)) + 1;
  char *knl = calloc(len, sizeof(char));
  snprintf(knl, len, knl_fmt, TOSTRING(TEST_TYPE), TOSTRING(TEST_TYPE));

  static int id = -1;
  const char *annotations[1] = {0},
             *clauses[3] = {"transform", "nomp-api-200:transform", 0};
  err = nomp_jit(&id, knl, annotations, clauses, 3, "a,b,N", NOMP_PTR,
                 sizeof(TEST_TYPE), a, NOMP_PTR, sizeof(TEST_TYPE), b,
                 NOMP_INTEGER, sizeof(int), &N);
  nomp_chk(err);
  free(knl);

  err = nomp_run(id, NOMP_PTR, a, NOMP_PTR, b, NOMP_INTEGER, &N, sizeof(int));
  nomp_chk(err);

  err = nomp_update(a, 0, 20, sizeof(TEST_TYPE), NOMP_FROM);
  nomp_chk(err);

#if defined(TEST_TOL)
  for (int i = 0; i < N; i++)
    nomp_assert(fabs(a[i] - 2 * b[i] - 1) < 1e-12);
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
