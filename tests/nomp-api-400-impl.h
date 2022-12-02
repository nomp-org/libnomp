#include "nomp-test.h"

#define nomp_api_400_aux TOKEN_PASTE(nomp_api_400_aux, TEST_SUFFIX)
int nomp_api_400_aux(TEST_TYPE *a, TEST_TYPE *z, int N) {
  const char *knl_fmt =
      "void foo(%s *a, %s *z, int N) {                        \n"
      "  for (int i = 0; i < N; i++) {                        \n"
      "    z[0] += a[i];                                      \n"
      "  }                                                    \n"
      "}                                                      \n";

  size_t len = strlen(knl_fmt) + 2 * strlen(TOSTRING(TEST_TYPE)) + 1;
  char *knl = tcalloc(char, len);
  snprintf(knl, len, knl_fmt, TOSTRING(TEST_TYPE), TOSTRING(TEST_TYPE),
           TOSTRING(TEST_TYPE));

  static int id = -1;
  const char *clauses[3] = {"reduce", "z", 0};
  int err = nomp_jit(&id, knl, clauses);
  nomp_chk(err);

  err =
      nomp_run(id, 3, "a", NOMP_PTR, sizeof(TEST_TYPE), a, "z", NOMP_PTR,
               sizeof(TEST_TYPE), z, "N", NOMP_INTEGER, sizeof(TEST_TYPE), &N);
  nomp_chk(err);

  tfree(knl);
  return 0;
}

#define nomp_api_400 TOKEN_PASTE(nomp_api_400, TEST_SUFFIX)
int nomp_api_400(const char *backend, int device, int platform) {
  int err = nomp_init(backend, platform, device);
  nomp_chk(err);

  int n = 10;
  TEST_TYPE a[10], sum;
  for (unsigned i = 0; i < n; i++)
    a[i] = n - i;

  err = nomp_update(a, 0, n, sizeof(TEST_TYPE), NOMP_TO);
  nomp_chk(err);

  nomp_api_400_aux(a, &sum, n);

  err = nomp_update(a, 0, n, sizeof(TEST_TYPE), NOMP_FROM);
  nomp_chk(err);

#if defined(TEST_TOL)
  for (unsigned i = 0; i < n; i++)
    nomp_assert(fabs(a[i] - n) < TEST_TOL);
#else
  for (unsigned i = 0; i < n; i++)
    nomp_assert(a[i] == n);
#endif

  err = nomp_update(a, 0, n, sizeof(TEST_TYPE), NOMP_FREE);
  nomp_chk(err);

  err = nomp_finalize();
  nomp_chk(err);

  return 0;
}
#undef nomp_api_400

#undef nomp_api_400_aux
