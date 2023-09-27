#include "nomp-test.h"

#define nomp_api_200_aux TOKEN_PASTE(nomp_api_200_aux, TEST_SUFFIX)
static int nomp_api_200_aux(const char *fmt, TEST_TYPE *a, TEST_TYPE *b,
                            int n) {
  nomp_test_check(nomp_update(a, 0, n, sizeof(TEST_TYPE), NOMP_TO));
  nomp_test_check(nomp_update(b, 0, n, sizeof(TEST_TYPE), NOMP_TO));

  int id = -1;
  const char *clauses[4] = {"transform", "nomp_api_100", "tile", 0};
  char *knl = generate_knl(fmt, 2, TOSTRING(TEST_TYPE), TOSTRING(TEST_TYPE));
  nomp_test_check(nomp_jit(&id, knl, clauses, 3, "a", sizeof(TEST_TYPE),
                           NOMP_PTR, "b", sizeof(TEST_TYPE), NOMP_PTR, "N",
                           sizeof(int), NOMP_INT));
  nomp_free(&knl);

  nomp_test_check(nomp_run(id, a, b, &n));

  nomp_test_check(nomp_sync());

  nomp_test_check(nomp_update(a, 0, n, sizeof(TEST_TYPE), NOMP_FROM));
  nomp_test_check(nomp_update(a, 0, n, sizeof(TEST_TYPE), NOMP_FREE));
  nomp_test_check(nomp_update(b, 0, n, sizeof(TEST_TYPE), NOMP_FREE));

  return 0;
}

#define nomp_api_200_add TOKEN_PASTE(nomp_api_200_add, TEST_SUFFIX)
static int nomp_api_200_add(unsigned n) {
  nomp_test_assert(n <= TEST_MAX_SIZE);

  TEST_TYPE a[TEST_MAX_SIZE], b[TEST_MAX_SIZE];
  for (unsigned i = 0; i < n; i++)
    a[i] = n - i, b[i] = i;

  const char *knl_fmt =
      "void foo(%s *a, %s *b, int N) {                        \n"
      "  for (int i = 0; i < N; i++)                          \n"
      "    a[i] += b[i];                                      \n"
      "}                                                      \n";
  nomp_api_200_aux(knl_fmt, a, b, n);

#if defined(TEST_TOL)
  for (unsigned i = 0; i < n; i++)
    nomp_test_assert(fabs(a[i] - n) < TEST_TOL);
#else
  for (unsigned i = 0; i < n; i++)
    nomp_test_assert(a[i] == (TEST_TYPE)n);
#endif

  return 0;
}
#undef nomp_api_200_add

#define nomp_api_200_ui_aux TOKEN_PASTE(nomp_api_200_ui_aux, TEST_SUFFIX)
static int nomp_api_200_ui_aux(const char *fmt, TEST_TYPE *a, TEST_TYPE *b,
                               unsigned n) {
  nomp_test_check(nomp_update(a, 0, n, sizeof(TEST_TYPE), NOMP_TO));
  nomp_test_check(nomp_update(b, 0, n, sizeof(TEST_TYPE), NOMP_TO));

  int id = -1;
  const char *clauses[4] = {"transform", "nomp_api_100", "tile", 0};
  char *knl = generate_knl(fmt, 2, TOSTRING(TEST_TYPE), TOSTRING(TEST_TYPE));
  nomp_test_check(nomp_jit(&id, knl, clauses, 3, "a", sizeof(TEST_TYPE),
                           NOMP_PTR, "b", sizeof(TEST_TYPE), NOMP_PTR, "N",
                           sizeof(unsigned), NOMP_UINT));
  nomp_free(&knl);

  nomp_test_check(nomp_run(id, a, b, &n));

  nomp_test_check(nomp_sync());

  nomp_test_check(nomp_update(a, 0, n, sizeof(TEST_TYPE), NOMP_FROM));
  nomp_test_check(nomp_update(a, 0, n, sizeof(TEST_TYPE), NOMP_FREE));
  nomp_test_check(nomp_update(b, 0, n, sizeof(TEST_TYPE), NOMP_FREE));

  return 0;
}

#define nomp_api_200_add_ui TOKEN_PASTE(nomp_api_200_add_ui, TEST_SUFFIX)
static int nomp_api_200_add_ui(unsigned n) {
  nomp_test_assert(n <= TEST_MAX_SIZE);

  TEST_TYPE a[TEST_MAX_SIZE], b[TEST_MAX_SIZE];
  for (unsigned i = 0; i < n; i++)
    a[i] = n - i, b[i] = i;

  const char *knl_fmt =
      "void foo(%s *a, %s *b, unsigned N) {                   \n"
      "  for (int i = 0; i < N; i++)                          \n"
      "    a[i] += b[i];                                      \n"
      "}                                                      \n";
  nomp_api_200_ui_aux(knl_fmt, a, b, n);

#if defined(TEST_TOL)
  for (unsigned i = 0; i < n; i++)
    nomp_test_assert(fabs(a[i] - n) < TEST_TOL);
#else
  for (unsigned i = 0; i < n; i++)
    nomp_test_assert(a[i] == (TEST_TYPE)n);
#endif

  return 0;
}
#undef nomp_api_200_add_ui

#define nomp_api_200_sub TOKEN_PASTE(nomp_api_200_sub, TEST_SUFFIX)
static int nomp_api_200_sub(unsigned n) {
  nomp_test_assert(n <= TEST_MAX_SIZE && n > 0);

  TEST_TYPE a[TEST_MAX_SIZE], b[TEST_MAX_SIZE];
  for (unsigned i = 0; i < n; i++)
    a[i] = n + i, b[i] = i;

  const char *knl_fmt =
      "void foo(%s *a, %s *b, int N) {                        \n"
      "  for (int i = 0; i < N; i++)                          \n"
      "    a[i] -= b[i] + 1;                                  \n"
      "}                                                      \n";
  nomp_api_200_aux(knl_fmt, a, b, n);

#if defined(TEST_TOL)
  for (unsigned i = 0; i < n; i++)
    nomp_test_assert(fabs(a[i] - n + 1) < TEST_TOL);
#else
  for (unsigned i = 0; i < n; i++) {
    nomp_test_assert(a[i] == (TEST_TYPE)n - 1);
  }
#endif

  return 0;
}
#undef nomp_api_200_sub

#define nomp_api_200_mul1 TOKEN_PASTE(nomp_api_200_mul1, TEST_SUFFIX)
static int nomp_api_200_mul1(unsigned n) {
  nomp_test_assert(n <= TEST_MAX_SIZE);

  TEST_TYPE a[TEST_MAX_SIZE], b[TEST_MAX_SIZE];
  for (unsigned i = 0; i < n; i++)
    a[i] = n - i, b[i] = i;

  const char *knl_fmt =
      "void foo(%s *a, %s *b, int N) {                        \n"
      "  for (int i = 0; i < N; i++)                          \n"
      "    a[i] = a[i] * b[i];                                \n"
      "}                                                      \n";
  nomp_api_200_aux(knl_fmt, a, b, n);

#if defined(TEST_TOL)
  for (unsigned i = 0; i < n; i++)
    nomp_test_assert(fabs(a[i] - (n - i) * i) < TEST_TOL);
#else
  for (unsigned i = 0; i < n; i++)
    nomp_test_assert(a[i] == (TEST_TYPE)((n - i) * i));
#endif

  return 0;
}
#undef nomp_api_200_mul1

#define nomp_api_200_mul2 TOKEN_PASTE(nomp_api_200_mul2, TEST_SUFFIX)
static int nomp_api_200_mul2(unsigned n) {
  nomp_test_assert(n <= TEST_MAX_SIZE);

  TEST_TYPE a[TEST_MAX_SIZE], b[TEST_MAX_SIZE];
  for (unsigned i = 0; i < n; i++)
    a[i] = n - i, b[i] = i;

  const char *knl_fmt =
      "void foo(%s *a, %s *b, int N) {                        \n"
      "  for (int i = 0; i < N; i++)                          \n"
      "    a[i] *= b[i] + 1;                                  \n"
      "}                                                      \n";
  nomp_api_200_aux(knl_fmt, a, b, n);

#if defined(TEST_TOL)
  for (unsigned i = 0; i < n; i++)
    nomp_test_assert(fabs(a[i] - (n - i) * (i + 1)) < TEST_TOL);
#else
  for (unsigned i = 0; i < n; i++)
    nomp_test_assert(a[i] == (TEST_TYPE)((n - i) * (i + 1)));
#endif

  return 0;
}
#undef nomp_api_200_mul2

#define nomp_api_200_square TOKEN_PASTE(nomp_api_200_square, TEST_SUFFIX)
static int nomp_api_200_square(unsigned n) {
  nomp_test_assert(n <= TEST_MAX_SIZE);

  TEST_TYPE a[TEST_MAX_SIZE], b[TEST_MAX_SIZE];
  for (unsigned i = 0; i < n; i++)
    a[i] = n - i, b[i] = i;

  const char *knl_fmt =
      "void foo(%s *a, %s *b, int N) {                        \n"
      "  for (int i = 0; i < N; i++)                          \n"
      "    a[i] = a[i] * a[i] + b[i] * b[i];                  \n"
      "}                                                      \n";
  nomp_api_200_aux(knl_fmt, a, b, n);

#if defined(TEST_TOL)
  for (unsigned i = 0; i < n; i++)
    nomp_test_assert(fabs(a[i] - (n - i) * (n - i) - i * i) < TEST_TOL);
#else
  for (unsigned i = 0; i < n; i++)
    nomp_test_assert(a[i] == (TEST_TYPE)((n - i) * (n - i) + i * i));
#endif

  return 0;
}
#undef nomp_api_200_square

#define nomp_api_200_saxpy TOKEN_PASTE(nomp_api_200_saxpy, TEST_SUFFIX)
static int nomp_api_200_saxpy(unsigned n) {
  nomp_test_assert(n <= TEST_MAX_SIZE);

  TEST_TYPE a[TEST_MAX_SIZE] = {0}, b[TEST_MAX_SIZE] = {1, 2, 3, 4, 5};

  const char *knl_fmt =
      "void foo(%s *a, %s *b, int N) {                        \n"
      "  for (int i = 0; i < N; i++)                          \n"
      "    a[i] = 2 * b[i] + 1;                               \n"
      "}                                                      \n";
  nomp_api_200_aux(knl_fmt, a, b, n);

#if defined(TEST_TOL)
  for (unsigned i = 0; i < n; i++)
    nomp_test_assert(fabs(a[i] - 2 * b[i] - 1) < TEST_TOL);
#else
  for (unsigned i = 0; i < n; i++)
    nomp_test_assert(a[i] == (TEST_TYPE)(2 * b[i] + 1));
#endif

  return 0;
}
#undef nomp_api_200_saxpy
#undef nomp_api_200_aux
