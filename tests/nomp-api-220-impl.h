#include "nomp-test.h"

#define nomp_api_220_aux TOKEN_PASTE(nomp_api_220_aux, TEST_SUFFIX)
static int nomp_api_220_aux(const char *knl_fmt, TEST_TYPE *a, TEST_TYPE *b,
                            TEST_TYPE *c, int n) {
  int err = nomp_update(a, 0, n, sizeof(TEST_TYPE), NOMP_TO);
  nomp_test_chk(err);
  err = nomp_update(b, 0, n, sizeof(TEST_TYPE), NOMP_TO);
  nomp_test_chk(err);
  err = nomp_update(c, 0, n, sizeof(TEST_TYPE), NOMP_TO);
  nomp_test_chk(err);

  const char *clauses[4] = {"transform", "nomp-api-220", "transform", 0};
  int id = -1;
  err = create_knl(&id, knl_fmt, clauses, 3, TOSTRING(TEST_TYPE),
                   TOSTRING(TEST_TYPE), TOSTRING(TEST_TYPE));
  nomp_test_chk(err);

  err = nomp_run(id, 4, "a", NOMP_PTR, sizeof(TEST_TYPE), a, "b", NOMP_PTR,
                 sizeof(TEST_TYPE), b, "c", NOMP_PTR, sizeof(TEST_TYPE), c, "N",
                 NOMP_INT, sizeof(int), &n);
  nomp_test_chk(err);

  err =nomp_sync();
  nomp_test_chk(err);

  err = nomp_update(a, 0, n, sizeof(TEST_TYPE), NOMP_FROM);
  nomp_test_chk(err);
  err = nomp_update(a, 0, n, sizeof(TEST_TYPE), NOMP_FREE);
  nomp_test_chk(err);
  err = nomp_update(b, 0, n, sizeof(TEST_TYPE), NOMP_FREE);
  nomp_test_chk(err);
  err = nomp_update(c, 0, n, sizeof(TEST_TYPE), NOMP_FREE);
  nomp_test_chk(err);

  return 0;
}

#define nomp_api_220_add TOKEN_PASTE(nomp_api_220_add, TEST_SUFFIX)
static int nomp_api_220_add(int n) {
  nomp_test_assert(n <= 20);

  TEST_TYPE a[20], b[20], c[20];
  for (unsigned i = 0; i < n; i++)
    a[i] = n - i, b[i] = i, c[i] = 5;

  const char *knl_fmt =
      "void foo(%s *a, %s *b, %s *c,int N) {                  \n"
      "  for (int i = 0; i < N; i++)                          \n"
      "    a[i] = a[i] + b[i] + c[i];                         \n"
      "}                                                      \n";
  nomp_api_220_aux(knl_fmt, a, b, c, n);

#if defined(TEST_TOL)
  for (unsigned i = 0; i < n; i++)
    nomp_test_assert(fabs(a[i] - n - 5) < TEST_TOL);
#else
  for (unsigned i = 0; i < n; i++)
    nomp_test_assert(a[i] == n + 5);
#endif

  return 0;
}
#undef nomp_api_220_add

#define nomp_api_220_mul TOKEN_PASTE(nomp_api_220_mul, TEST_SUFFIX)
static int nomp_api_220_mul(int n) {
  nomp_test_assert(n <= 20);

  TEST_TYPE a[20], b[20], c[20];
  for (unsigned i = 0; i < n; i++)
    a[i] = n - i, b[i] = i, c[i] = 5;

  const char *knl_fmt =
      "void foo(%s *a, %s *b, %s *c, int N) {                 \n"
      "  for (int i = 0; i < N; i++)                          \n"
      "    a[i] = a[i] * b[i] * c[i];                         \n"
      "}                                                      \n";
  nomp_api_220_aux(knl_fmt, a, b, c, n);

#if defined(TEST_TOL)
  for (unsigned i = 0; i < n; i++)
    nomp_test_assert(fabs(a[i] - 5 * (n - i) * i) < TEST_TOL);
#else
  for (unsigned i = 0; i < n; i++)
    nomp_test_assert(a[i] == 5 * (n - i) * i);
#endif

  return 0;
}
#undef nomp_api_220_mul

#define nomp_api_220_mul_sum TOKEN_PASTE(nomp_api_220_mul_sum, TEST_SUFFIX)
static int nomp_api_220_mul_sum(int n) {
  nomp_test_assert(n <= 20);

  TEST_TYPE a[20], b[20], c[20];
  for (unsigned i = 0; i < n; i++)
    a[i] = n - i, b[i] = i, c[i] = 5;

  const char *knl_fmt =
      "void foo(%s *a, %s *b, %s *c, int N) {                 \n"
      "  for (int i = 0; i < N; i++)                          \n"
      "    a[i] = a[i] * b[i] + c[i];                         \n"
      "}                                                      \n";
  nomp_api_220_aux(knl_fmt, a, b, c, n);

#if defined(TEST_TOL)
  for (unsigned i = 0; i < n; i++)
    nomp_test_assert(fabs(a[i] - (n - i) * i - 5) < TEST_TOL);
#else
  for (unsigned i = 0; i < n; i++)
    nomp_test_assert(a[i] == (n - i) * i + 5);
#endif

  return 0;
}
#undef nomp_api_220_mul_sum

#define nomp_api_220_linear TOKEN_PASTE(nomp_api_220_linear, TEST_SUFFIX)
static int nomp_api_220_linear(int n) {
  nomp_test_assert(n <= 20);

  TEST_TYPE a[20], b[20], c[20];
  for (unsigned i = 0; i < n; i++)
    a[i] = n - i, b[i] = i, c[i] = 5;

  const char *knl_fmt =
      "void foo(%s *a, %s *b, %s *c, int N) {                 \n"
      "  for (int i = 0; i < N; i++)                          \n"
      "    a[i] = a[i] + 3 * b[i] + 2 * c[i];                 \n"
      "}                                                      \n";
  nomp_api_220_aux(knl_fmt, a, b, c, n);

#if defined(TEST_TOL)
  for (unsigned i = 0; i < n; i++)
    nomp_test_assert(fabs(a[i] - n - 2 * i - 10) < TEST_TOL);
#else
  for (unsigned i = 0; i < n; i++)
    nomp_test_assert(a[i] == n + 2 * i + 10);
#endif

  return 0;
}
#undef nomp_api_220_linear
#undef nomp_api_220_aux
