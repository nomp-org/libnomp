#include "nomp-test.h"

#define nomp_api_400_aux TOKEN_PASTE(nomp_api_400_aux, TEST_SUFFIX)
int nomp_api_400_aux(TEST_TYPE *a, TEST_TYPE *z, int N) {
  const char *KNL_FMT =
      "void foo(%s *a, %s *z, int N) {                        \n"
      "  for (int i = 0; i < N; i++) {                        \n"
      "    z[0] += a[i];                                      \n"
      "  }                                                    \n"
      "}                                                      \n";

  const char *TYPE_STR = TOSTRING(TEST_TYPE);
  size_t len = strlen(KNL_FMT) + 2 * strlen(TYPE_STR) + 1;
  char *knl = tcalloc(char, len);
  snprintf(knl, len, KNL_FMT, TYPE_STR, TYPE_STR);

  static int id = -1;
  const char *clauses[1] = {0};
  int err = nomp_jit(&id, knl, clauses, 3, "a", NOMP_PTR, sizeof(TEST_TYPE),
                     "z", NOMP_PTR | NOMP_ATTR_REDN, sizeof(TEST_TYPE), "N",
                     NOMP_INT, sizeof(int));
  tfree(knl);
  nomp_chk(err);

  err = nomp_run(id, a, z, &N);
  nomp_chk(err);

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
