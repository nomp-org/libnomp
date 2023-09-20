#include "nomp-test.h"

#define nomp_api_500_sum_aux TOKEN_PASTE(nomp_api_500_sum_aux, TEST_SUFFIX)
static int nomp_api_500_sum_aux(const char *fmt, const char **clauses,
                                TEST_TYPE *a, int n) {
  int id = -1;
  char *knl = generate_knl(fmt, 1, TOSTRING(TEST_TYPE));
  nomp_test_check(nomp_jit(&id, knl, clauses, 2, "a", sizeof(TEST_TYPE),
                           TEST_NOMP_TYPE, "N", sizeof(int), NOMP_INT));
  nomp_free(&knl);

  nomp_test_check(nomp_run(id, a, &n));

  return 0;
}

#define nomp_api_500_sum_const TOKEN_PASTE(nomp_api_500_sum_const, TEST_SUFFIX)
static int nomp_api_500_sum_const(unsigned N) {
  nomp_test_assert(N <= TEST_MAX_SIZE);

  TEST_TYPE a[TEST_MAX_SIZE] = {0};
  const char *knl_fmt =
      "void foo(%s *a, int N) {                                        \n"
      "  for (int i = 0; i < N; i++) {                                 \n"
      "    a[0] += 1;                                                  \n"
      "  }                                                             \n"
      "}                                                               \n";
  const char *clauses[4] = {"reduce", "a", "+", NULL};
  nomp_api_500_sum_aux(knl_fmt, clauses, a, N);

#if defined(TEST_TOL)
  nomp_test_assert(fabs(a[0] - N) < TEST_TOL);
#else
  nomp_test_assert(a[0] == (TEST_TYPE)N);
#endif

  return 0;
}
#undef nomp_api_500_sum_const

#define nomp_api_500_sum_var TOKEN_PASTE(nomp_api_500_sum_var, TEST_SUFFIX)
static int nomp_api_500_sum_var(unsigned N) {
  nomp_test_assert(N <= TEST_MAX_SIZE && N > 0);

  TEST_TYPE a[TEST_MAX_SIZE] = {0};
  const char *knl_fmt =
      "void foo(%s *a, int N) {                                        \n"
      "  for (int i = 0; i < N; i++) {                                 \n"
      "    a[0] += i;                                                  \n"
      "  }                                                             \n"
      "}                                                               \n";
  const char *clauses[4] = {"reduce", "a", "+", NULL};
  nomp_api_500_sum_aux(knl_fmt, clauses, a, N);

#if defined(TEST_TOL)
  nomp_test_assert(fabs(a[0] - (N - 1) * N / 2) < TEST_TOL);
#else
  nomp_test_assert(a[0] == (TEST_TYPE)((N - 1) * N / 2));
#endif

  return 0;
}
#undef nomp_api_500_sum_var
#undef nomp_api_500_sum_aux

#define nomp_api_500_sum_array_aux                                             \
  TOKEN_PASTE(nomp_api_500_sum_array_aux, TEST_SUFFIX)
static int nomp_api_500_sum_array_aux(const char *fmt, const char **clauses,
                                      TEST_TYPE *a, int n, TEST_TYPE *sum) {
  nomp_test_check(nomp_update(a, 0, n, sizeof(TEST_TYPE), NOMP_TO));

  int id = -1;
  char *knl = generate_knl(fmt, 2, TOSTRING(TEST_TYPE), TOSTRING(TEST_TYPE));
  nomp_test_check(nomp_jit(&id, knl, clauses, 3, "a", sizeof(TEST_TYPE),
                           NOMP_PTR, "N", sizeof(int), NOMP_INT, "sum",
                           sizeof(TEST_TYPE), TEST_NOMP_TYPE));
  nomp_free(&knl);

  nomp_test_check(nomp_run(id, a, &n, sum));
  nomp_test_check(nomp_sync());
  nomp_test_check(nomp_update(a, 0, n, sizeof(TEST_TYPE), NOMP_FREE));

  return 0;
}

#define nomp_api_500_sum_array TOKEN_PASTE(nomp_api_500_sum_array, TEST_SUFFIX)
static int nomp_api_500_sum_array(unsigned N) {
  nomp_test_assert(N <= TEST_MAX_SIZE && N > 0);

  TEST_TYPE a[TEST_MAX_SIZE];
  for (unsigned i = 0; i < N; i++)
    a[i] = i;

  const char *knl_fmt =
      "void foo(%s *a, int N, %s *sum) {                               \n"
      "  for (int i = 0; i < N; i++) {                                 \n"
      "    sum[0] += a[i];                                             \n"
      "  }                                                             \n"
      "}                                                               \n";
  const char *clauses[4] = {"reduce", "sum", "+", NULL};
  TEST_TYPE sum;
  nomp_api_500_sum_array_aux(knl_fmt, clauses, a, N, &sum);

#if defined(TEST_TOL)
  nomp_test_assert(fabs(sum - (N - 1) * N / 2) < TEST_TOL);
#else
  nomp_test_assert(sum == (TEST_TYPE)((N - 1) * N / 2));
#endif

  return 0;
}
#undef nomp_api_500_sum_array

#define nomp_api_500_condition TOKEN_PASTE(nomp_api_500_condition, TEST_SUFFIX)
static int nomp_api_500_condition(unsigned N) {
  nomp_test_assert(N <= TEST_MAX_SIZE);

  TEST_TYPE a[TEST_MAX_SIZE];
  const unsigned mid_point = N / 2;
  for (unsigned i = 0; i < mid_point; ++i)
    a[i] = 0;
  for (unsigned i = mid_point; i < N; ++i)
    a[i] = i;

  const char *knl_fmt =
      "void foo(%s *a, int N, %s *sum) {                               \n"
      "  for (int i = 0; i < N; i++) {                                 \n"
      "    if (a[i] > 0)                                               \n"
      "      sum[0] += 1;                                              \n"
      "  }                                                             \n"
      "}                                                               \n";
  const char *clauses[4] = {"reduce", "sum", "+", NULL};
  TEST_TYPE sum;
  nomp_api_500_sum_array_aux(knl_fmt, clauses, a, N, &sum);

#if defined(TEST_TOL)
  nomp_test_assert(fabs(sum - mid_point) < TEST_TOL);
#else
  nomp_test_assert(sum == (TEST_TYPE)mid_point);
#endif

  return 0;
}
#undef nomp_api_500_condition
#undef nomp_api_500_sum_array_aux

#define nomp_api_500_dot_aux TOKEN_PASTE(nomp_api_500_dot_aux, TEST_SUFFIX)
static int nomp_api_500_dot_aux(const char *fmt, const char **clauses,
                                TEST_TYPE *a, TEST_TYPE *b, int n,
                                TEST_TYPE *total) {
  nomp_test_check(nomp_update(a, 0, n, sizeof(TEST_TYPE), NOMP_TO));
  nomp_test_check(nomp_update(b, 0, n, sizeof(TEST_TYPE), NOMP_TO));

  int id = -1;
  char *knl = generate_knl(fmt, 3, TOSTRING(TEST_TYPE), TOSTRING(TEST_TYPE),
                           TOSTRING(TEST_TYPE));
  nomp_test_check(nomp_jit(&id, knl, clauses, 4, "a", sizeof(TEST_TYPE *),
                           NOMP_PTR, "b", sizeof(TEST_TYPE), NOMP_PTR, "N",
                           sizeof(int), NOMP_INT, "total", sizeof(TEST_TYPE),
                           TEST_NOMP_TYPE));
  nomp_free(&knl);

  nomp_test_check(nomp_run(id, a, b, &n, total));
  nomp_test_check(nomp_sync());
  nomp_test_check(nomp_update(a, 0, n, sizeof(TEST_TYPE), NOMP_FREE));
  nomp_test_check(nomp_update(b, 0, n, sizeof(TEST_TYPE), NOMP_FREE));

  return 0;
}

#define nomp_api_500_dot TOKEN_PASTE(nomp_api_500_dot, TEST_SUFFIX)
static int nomp_api_500_dot(unsigned N) {
  nomp_test_assert(N <= TEST_MAX_SIZE && N > 0);

  TEST_TYPE a[TEST_MAX_SIZE], b[TEST_MAX_SIZE], total;
  for (unsigned i = 0; i < N; i++)
    a[i] = i, b[i] = i;

  const char *knl_fmt =
      "void foo(%s *a, %s *b, int N, %s *total) {                        \n"
      "  for (int i = 0; i < N; i++) {                                   \n"
      "    total[0] += a[i] * b[i];                                      \n"
      "  }                                                               \n"
      "}                                                                 \n";
  const char *clauses[4] = {"reduce", "total", "+", NULL};
  nomp_api_500_dot_aux(knl_fmt, clauses, a, b, N, &total);

#if defined(TEST_TOL)
  nomp_test_assert(fabs(total - N * (2 * N - 1) * (N - 1) / 6) < TEST_TOL);
#else
  nomp_test_assert(total == N * (2 * N - 1) * (N - 1) / 6);
#endif

  return 0;
}
#undef nomp_api_500_dot
#undef nomp_api_500_dot_aux

#define nomp_api_500_multiple_reductions_aux                                   \
  TOKEN_PASTE(nomp_api_500_multiple_reductions_aux, TEST_SUFFIX)
static int nomp_api_500_multiple_reductions_aux(const char *fmt, TEST_TYPE *a,
                                                int n, TEST_TYPE *total) {
  nomp_test_check(nomp_update(a, 0, n, sizeof(TEST_TYPE), NOMP_TO));

  int id = -1;
  const char *clauses[4] = {"reduce", "total", "+", NULL};
  char *knl = generate_knl(fmt, 2, TOSTRING(TEST_TYPE), TOSTRING(TEST_TYPE));
  nomp_test_check(nomp_jit(&id, knl, clauses, 3, "a", sizeof(TEST_TYPE *),
                           NOMP_PTR, "N", sizeof(int), NOMP_INT, "total",
                           sizeof(TEST_TYPE), TEST_NOMP_TYPE));
  nomp_free(&knl);

  nomp_test_check(nomp_run(id, a, &n, total));
  nomp_test_check(nomp_update(a, 0, n, sizeof(TEST_TYPE), NOMP_FREE));

  return 0;
}

#define nomp_api_500_multiple_reductions                                       \
  TOKEN_PASTE(nomp_api_500_multiple_reductions, TEST_SUFFIX)
static int nomp_api_500_multiple_reductions(unsigned N, unsigned iterations) {
  nomp_test_assert(N <= TEST_MAX_SIZE && N > 0);

  const char *knl_fmt =
      "void foo(%s *a, int N, %s *total) {                               \n"
      "  for (int i = 0; i < N; i++) {                                   \n"
      "    total[0] += a[i];                                             \n"
      "  }                                                               \n"
      "}                                                                 \n";

  TEST_TYPE a[TEST_MAX_SIZE];
  for (unsigned i = 1; i < iterations; ++i) {
    for (unsigned j = 0; j < N; j++)
      a[j] = i * j;

    TEST_TYPE total = 0;
    nomp_api_500_multiple_reductions_aux(knl_fmt, a, N, &total);

#if defined(TEST_TOL)
    nomp_test_assert(fabs(total - (N - 1) * N * i / 2) < TEST_TOL);
#else
    nomp_test_assert(total == (N - 1) * N * i / 2);
#endif
  }

  return 0;
}
#undef nomp_api_500_multiple_reductions
#undef nomp_api_500_multiple_reductions_aux
