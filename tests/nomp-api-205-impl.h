#include "nomp-test.h"

#define nomp_api_205_aux TOKEN_PASTE(nomp_api_205_aux, TEST_SUFFIX)
static int nomp_api_205_aux(const char *fmt, TEST_TYPE *a, TEST_TYPE *b,
                            TEST_TYPE *c, int n) {
  nomp_test_check(nomp_update(a, 0, n, sizeof(TEST_TYPE), NOMP_TO));
  nomp_test_check(nomp_update(b, 0, n, sizeof(TEST_TYPE), NOMP_TO));
  nomp_test_check(nomp_update(c, 0, n, sizeof(TEST_TYPE), NOMP_TO));

  int id = -1;
  const char *clauses[4] = {"transform", "nomp_api_205", "transform", 0};
  char *knl = generate_knl(fmt, 3, TOSTRING(TEST_TYPE), TOSTRING(TEST_TYPE),
                           TOSTRING(TEST_TYPE));
  nomp_test_check(nomp_jit(&id, knl, clauses, 4, "a", sizeof(TEST_TYPE),
                           NOMP_PTR, "b", sizeof(TEST_TYPE), NOMP_PTR, "c",
                           sizeof(TEST_TYPE), NOMP_PTR, "N", sizeof(int),
                           NOMP_INT));
  nomp_free(&knl);

  nomp_test_check(nomp_run(id, a, b, c, &n));
  nomp_test_check(nomp_sync());
  nomp_test_check(nomp_update(a, 0, n, sizeof(TEST_TYPE), NOMP_FROM));
  nomp_test_check(nomp_update(a, 0, n, sizeof(TEST_TYPE), NOMP_FREE));
  nomp_test_check(nomp_update(b, 0, n, sizeof(TEST_TYPE), NOMP_FREE));
  nomp_test_check(nomp_update(c, 0, n, sizeof(TEST_TYPE), NOMP_FREE));

  return 0;
}

#define nomp_api_205_add TOKEN_PASTE(nomp_api_205_add, TEST_SUFFIX)
static int nomp_api_205_add(unsigned n) {
  nomp_test_assert(n <= TEST_MAX_SIZE);

  TEST_TYPE a[TEST_MAX_SIZE], b[TEST_MAX_SIZE], c[TEST_MAX_SIZE];
  for (unsigned i = 0; i < n; i++)
    a[i] = n - i, b[i] = i, c[i] = 5;

  const char *knl_fmt =
      "void foo(%s *a, %s *b, %s *c,int N) {                  \n"
      "  for (int i = 0; i < N; i++)                          \n"
      "    a[i] = a[i] + b[i] + c[i];                         \n"
      "}                                                      \n";
  nomp_api_205_aux(knl_fmt, a, b, c, n);

#if defined(TEST_TOL)
  for (unsigned i = 0; i < n; i++)
    nomp_test_assert(fabs(a[i] - n - 5) < TEST_TOL);
#else
  for (unsigned i = 0; i < n; i++)
    nomp_test_assert(a[i] == (TEST_TYPE)(n + 5));
#endif

  return 0;
}
#undef nomp_api_205_add

#define nomp_api_205_mul TOKEN_PASTE(nomp_api_205_mul, TEST_SUFFIX)
static int nomp_api_205_mul(unsigned n) {
  nomp_test_assert(n <= TEST_MAX_SIZE);

  TEST_TYPE a[TEST_MAX_SIZE], b[TEST_MAX_SIZE], c[TEST_MAX_SIZE];
  for (unsigned i = 0; i < n; i++)
    a[i] = n - i, b[i] = i, c[i] = 5;

  const char *knl_fmt =
      "void foo(%s *a, %s *b, %s *c, int N) {                 \n"
      "  for (int i = 0; i < N; i++)                          \n"
      "    a[i] = a[i] * b[i] * c[i];                         \n"
      "}                                                      \n";
  nomp_api_205_aux(knl_fmt, a, b, c, n);

#if defined(TEST_TOL)
  for (unsigned i = 0; i < n; i++)
    nomp_test_assert(fabs(a[i] - 5 * (n - i) * i) < TEST_TOL);
#else
  for (unsigned i = 0; i < n; i++)
    nomp_test_assert(a[i] == (TEST_TYPE)(5 * (n - i) * i));
#endif

  return 0;
}
#undef nomp_api_205_mul

#define nomp_api_205_mul_sum TOKEN_PASTE(nomp_api_205_mul_sum, TEST_SUFFIX)
static int nomp_api_205_mul_sum(unsigned n) {
  nomp_test_assert(n <= TEST_MAX_SIZE);

  TEST_TYPE a[TEST_MAX_SIZE], b[TEST_MAX_SIZE], c[TEST_MAX_SIZE];
  for (unsigned i = 0; i < n; i++)
    a[i] = n - i, b[i] = i, c[i] = 5;

  const char *knl_fmt =
      "void foo(%s *a, %s *b, %s *c, int N) {                 \n"
      "  for (int i = 0; i < N; i++)                          \n"
      "    a[i] = a[i] * b[i] + c[i];                         \n"
      "}                                                      \n";
  nomp_api_205_aux(knl_fmt, a, b, c, n);

#if defined(TEST_TOL)
  for (unsigned i = 0; i < n; i++)
    nomp_test_assert(fabs(a[i] - (n - i) * i - 5) < TEST_TOL);
#else
  for (unsigned i = 0; i < n; i++)
    nomp_test_assert(a[i] == (TEST_TYPE)((n - i) * i + 5));
#endif

  return 0;
}
#undef nomp_api_205_mul_sum

#define nomp_api_205_linear TOKEN_PASTE(nomp_api_205_linear, TEST_SUFFIX)
static int nomp_api_205_linear(unsigned n) {
  nomp_test_assert(n <= TEST_MAX_SIZE);

  TEST_TYPE a[TEST_MAX_SIZE], b[TEST_MAX_SIZE], c[TEST_MAX_SIZE];
  for (unsigned i = 0; i < n; i++)
    a[i] = n - i, b[i] = i, c[i] = 5;

  const char *knl_fmt =
      "void foo(%s *a, %s *b, %s *c, int N) {                 \n"
      "  for (int i = 0; i < N; i++)                          \n"
      "    a[i] = a[i] + 3 * b[i] + 2 * c[i];                 \n"
      "}                                                      \n";
  nomp_api_205_aux(knl_fmt, a, b, c, n);

#if defined(TEST_TOL)
  for (unsigned i = 0; i < n; i++)
    nomp_test_assert(fabs(a[i] - n - 2 * i - 10) < TEST_TOL);
#else
  for (unsigned i = 0; i < n; i++)
    nomp_test_assert(a[i] == (TEST_TYPE)(n + 2 * i + 10));
#endif

  return 0;
}
#undef nomp_api_205_linear
#undef nomp_api_205_aux
