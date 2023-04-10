#include "nomp-test.h"

#define nomp_api_210_aux TOKEN_PASTE(nomp_api_210_aux, TEST_SUFFIX)
static int nomp_api_210_aux(const char *knl_fmt, TEST_TYPE *a, TEST_TYPE *b,
                            int n) {
  nomp_test_chk(nomp_update(a, 0, n, sizeof(TEST_TYPE), NOMP_TO));
  nomp_test_chk(nomp_update(b, 0, n, sizeof(TEST_TYPE), NOMP_TO));

  const char *clauses[4] = {"transform", "nomp-api-210", "transform", 0};
  int id = -1;
  nomp_test_chk(create_knl(&id, knl_fmt, clauses, 2, TOSTRING(TEST_TYPE),
                           TOSTRING(TEST_TYPE)));

  nomp_test_chk(nomp_run(id, 3, "a", NOMP_PTR, sizeof(TEST_TYPE), a, "b",
                         NOMP_PTR, sizeof(TEST_TYPE), b, "N", NOMP_INT,
                         sizeof(int), &n));

  nomp_test_chk(nomp_sync());

  nomp_test_chk(nomp_update(a, 0, n, sizeof(TEST_TYPE), NOMP_FROM));
  nomp_test_chk(nomp_update(a, 0, n, sizeof(TEST_TYPE), NOMP_FREE));
  nomp_test_chk(nomp_update(b, 0, n, sizeof(TEST_TYPE), NOMP_FREE));

  return 0;
}

#define nomp_api_210_add TOKEN_PASTE(nomp_api_210_add, TEST_SUFFIX)
static int nomp_api_210_add(int n) {
  nomp_test_assert(n <= 20);

  TEST_TYPE a[20], b[20];
  for (unsigned i = 0; i < n; i++)
    a[i] = n - i, b[i] = i;

  const char *knl_fmt =
      "void foo(%s *a, %s *b, int N) {                        \n"
      "  for (int i = 0; i < N; i++)                          \n"
      "    a[i] += b[i];                                      \n"
      "}                                                      \n";
  nomp_api_210_aux(knl_fmt, a, b, n);

#if defined(TEST_TOL)
  for (unsigned i = 0; i < n; i++)
    nomp_test_assert(fabs(a[i] - n) < TEST_TOL);
#else
  for (unsigned i = 0; i < n; i++)
    nomp_test_assert(a[i] == n);
#endif

  return 0;
}
#undef nomp_api_210_add

#define nomp_api_210_sub TOKEN_PASTE(nomp_api_210_sub, TEST_SUFFIX)
static int nomp_api_210_sub(int n) {
  nomp_test_assert(n <= 20);

  TEST_TYPE a[20], b[20];
  for (unsigned i = 0; i < n; i++)
    a[i] = n + i, b[i] = i;

  const char *knl_fmt =
      "void foo(%s *a, %s *b, int N) {                        \n"
      "  for (int i = 0; i < N; i++)                          \n"
      "    a[i] -= b[i] + 1;                                  \n"
      "}                                                      \n";
  nomp_api_210_aux(knl_fmt, a, b, n);

#if defined(TEST_TOL)
  for (unsigned i = 0; i < n; i++)
    nomp_test_assert(fabs(a[i] - n + 1) < TEST_TOL);
#else
  for (unsigned i = 0; i < n; i++) {
    nomp_test_assert(a[i] == n - 1);
  }
#endif

  return 0;
}
#undef nomp_api_210_sub

#define nomp_api_210_mul_sum TOKEN_PASTE(nomp_api_210_mul_sum, TEST_SUFFIX)
static int nomp_api_210_mul_sum(int n) {
  nomp_test_assert(n <= 20);

  TEST_TYPE a[20], b[20];
  for (unsigned i = 0; i < n; i++)
    a[i] = n - i, b[i] = i;

  const char *knl_fmt =
      "void foo(%s *a, %s *b, int N) {                        \n"
      "  for (int i = 0; i < N; i++)                          \n"
      "    a[i] *= b[i] + 1;                                  \n"
      "}                                                      \n";
  nomp_api_210_aux(knl_fmt, a, b, n);

#if defined(TEST_TOL)
  for (unsigned i = 0; i < n; i++)
    nomp_test_assert(fabs(a[i] - (n - i) * (i + 1)) < TEST_TOL);
#else
  for (unsigned i = 0; i < n; i++)
    nomp_test_assert(a[i] == (n - i) * (i + 1));
#endif

  return 0;
}
#undef nomp_api_210_mul_sum

#define nomp_api_210_mul TOKEN_PASTE(nomp_api_210_mul, TEST_SUFFIX)
static int nomp_api_210_mul(int n) {
  nomp_test_assert(n <= 20);

  TEST_TYPE a[20], b[20];
  for (unsigned i = 0; i < n; i++)
    a[i] = n - i, b[i] = i;

  const char *knl_fmt =
      "void foo(%s *a, %s *b, int N) {                        \n"
      "  for (int i = 0; i < N; i++)                          \n"
      "    a[i] = a[i] * b[i];                                \n"
      "}                                                      \n";
  nomp_api_210_aux(knl_fmt, a, b, n);

#if defined(TEST_TOL)
  for (unsigned i = 0; i < n; i++)
    nomp_test_assert(fabs(a[i] - (n - i) * i) < TEST_TOL);
#else
  for (unsigned i = 0; i < n; i++)
    nomp_test_assert(a[i] == (n - i) * i);
#endif

  return 0;
}
#undef nomp_api_210_mul

#define nomp_api_210_square TOKEN_PASTE(nomp_api_210_square, TEST_SUFFIX)
static int nomp_api_210_square(int n) {
  nomp_test_assert(n <= 20);

  TEST_TYPE a[20], b[20];
  for (unsigned i = 0; i < n; i++)
    a[i] = n - i, b[i] = i;

  const char *knl_fmt =
      "void foo(%s *a, %s *b, int N) {                        \n"
      "  for (int i = 0; i < N; i++)                          \n"
      "    a[i] = a[i] * a[i] + b[i] * b[i];                  \n"
      "}                                                      \n";
  nomp_api_210_aux(knl_fmt, a, b, n);

#if defined(TEST_TOL)
  for (unsigned i = 0; i < n; i++)
    nomp_test_assert(fabs(a[i] - (n - i) * (n - i) - i * i) < TEST_TOL);
#else
  for (unsigned i = 0; i < n; i++)
    nomp_test_assert(a[i] == (n - i) * (n - i) + i * i);
#endif

  return 0;
}
#undef nomp_api_210_square

#define nomp_api_210_linear TOKEN_PASTE(nomp_api_210_linear, TEST_SUFFIX)
static int nomp_api_210_linear(int n) {
  nomp_test_assert(n <= 20);

  TEST_TYPE a[20] = {0}, b[20] = {1, 2, 3, 4, 5};

  const char *knl_fmt =
      "void foo(%s *a, %s *b, int N) {                        \n"
      "  for (int i = 0; i < N; i++)                          \n"
      "    a[i] = 2 * b[i] + 1;                               \n"
      "}                                                      \n";
  nomp_api_210_aux(knl_fmt, a, b, n);

#if defined(TEST_TOL)
  for (int i = 0; i < n; i++)
    nomp_test_assert(fabs(a[i] - 2 * b[i] - 1) < TEST_TOL);
#else
  for (int i = 0; i < n; i++)
    nomp_test_assert(a[i] == 2 * b[i] + 1);
#endif

  return 0;
}
#undef nomp_api_210_linear
#undef nomp_api_210_aux
