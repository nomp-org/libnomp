#include "nomp-test.h"

#define nomp_api_500_sum_const_aux                                             \
  TOKEN_PASTE(nomp_api_500_sum_const_aux, TEST_SUFFIX)
static int nomp_api_500_sum_const_aux(const char *fmt, const char **clauses,
                                      TEST_TYPE *a, int n) {
  nomp_test_chk(nomp_update(a, 0, n, sizeof(TEST_TYPE), NOMP_TO));

  int id = -1;
  char *knl = generate_knl(fmt, 1, TOSTRING(TEST_TYPE));
  nomp_test_chk(
      nomp_jit(&id, knl, clauses, 1, "a", sizeof(TEST_TYPE), TEST_NOMP_TYPE));
  nomp_free(&knl);

  nomp_test_chk(nomp_run(id, a, &n));
  nomp_test_chk(nomp_sync());
  nomp_test_chk(nomp_update(a, 0, n, sizeof(TEST_TYPE), NOMP_FREE));

  return 0;
}

#define nomp_api_500_sum_const TOKEN_PASTE(nomp_api_500_sum_const, TEST_SUFFIX)
static int nomp_api_500_sum_const(void) {
  TEST_TYPE a;
  const int N = 10;
  const char *knl_fmt =
      "void foo(%s *a) {                                               \n"
      "  for (int i = 0; i < 10; i++) {                                \n"
      "    a[0] += i;                                                  \n"
      "  }                                                             \n"
      "}                                                               \n";
  const char *clauses[4] = {"reduce", "a", "+", NULL};
  nomp_api_500_sum_const_aux(knl_fmt, clauses, &a, N);

#if defined(TEST_TOL)
  nomp_test_assert(fabs(a - (N - 1) * N / 2) < TEST_TOL);
#else
  nomp_test_assert(a == (N - 1) * N / 2);
#endif

  return 0;
}
#undef nomp_api_500_sum_const
#undef nomp_api_500_sum_const_aux

#define nomp_api_500_sum_aux TOKEN_PASTE(nomp_api_500_sum_aux, TEST_SUFFIX)
static int nomp_api_500_sum_aux(const char *fmt, const char **clauses,
                                TEST_TYPE *a, int n) {
  nomp_test_chk(nomp_update(a, 0, n, sizeof(TEST_TYPE), NOMP_TO));

  int id = -1;
  char *knl = generate_knl(fmt, 1, TOSTRING(TEST_TYPE));
  nomp_test_chk(nomp_jit(&id, knl, clauses, 2, "a", sizeof(TEST_TYPE),
                         TEST_NOMP_TYPE, "N", sizeof(int), NOMP_INT));
  nomp_free(&knl);

  nomp_test_chk(nomp_run(id, a, &n));
  nomp_test_chk(nomp_sync());
  nomp_test_chk(nomp_update(a, 0, n, sizeof(TEST_TYPE), NOMP_FREE));

  return 0;
}

#define nomp_api_500_sum TOKEN_PASTE(nomp_api_500_sum, TEST_SUFFIX)
static int nomp_api_500_sum(int N) {
  TEST_TYPE a;
  const char *knl_fmt =
      "void foo(%s *a, int N) {                                        \n"
      "  for (int i = 0; i < N; i++) {                                 \n"
      "    a[0] += i;                                                  \n"
      "  }                                                             \n"
      "}                                                               \n";
  const char *clauses[4] = {"reduce", "a", "+", NULL};
  nomp_api_500_sum_aux(knl_fmt, clauses, &a, N);

#if defined(TEST_TOL)
  nomp_test_assert(fabs(a - (N - 1) * N / 2) < TEST_TOL);
#else
  nomp_test_assert(a == (N - 1) * N / 2);
#endif

  return 0;
}
#undef nomp_api_500_sum
#undef nomp_api_500_sum_aux

#define nomp_api_500_sum_array_aux                                             \
  TOKEN_PASTE(nomp_api_500_sum_array_aux, TEST_SUFFIX)
static int nomp_api_500_sum_array_aux(const char *fmt, const char **clauses,
                                      TEST_TYPE *a, int n, TEST_TYPE *sum) {
  nomp_test_chk(nomp_update(a, 0, n, sizeof(TEST_TYPE), NOMP_TO));

  int id = -1;
  char *knl = generate_knl(fmt, 2, TOSTRING(TEST_TYPE), TOSTRING(TEST_TYPE));
  nomp_test_chk(nomp_jit(&id, knl, clauses, 3, "a", sizeof(TEST_TYPE), NOMP_PTR,
                         "N", sizeof(int), NOMP_INT, "sum", sizeof(TEST_TYPE),
                         TEST_NOMP_TYPE));
  nomp_free(&knl);

  nomp_test_chk(nomp_run(id, a, &n, sum));
  nomp_test_chk(nomp_sync());
  nomp_test_chk(nomp_update(a, 0, n, sizeof(TEST_TYPE), NOMP_FREE));

  return 0;
}

#define nomp_api_500_sum_array TOKEN_PASTE(nomp_api_500_sum_array, TEST_SUFFIX)
static int nomp_api_500_sum_array(int N) {
  nomp_test_assert(N <= 10);

  TEST_TYPE a[10], sum;
  for (unsigned i = 0; i < N; i++)
    a[i] = i;

  const char *knl_fmt =
      "void foo(%s *a, int N, %s *sum) {                               \n"
      "  for (int i = 0; i < N; i++) {                                 \n"
      "    sum[0] += a[i];                                             \n"
      "  }                                                             \n"
      "}                                                               \n";
  const char *clauses[4] = {"reduce", "sum", "+", NULL};
  nomp_api_500_sum_array_aux(knl_fmt, clauses, a, N, &sum);

#if defined(TEST_TOL)
  nomp_test_assert(fabs(sum - (N - 1) * N / 2) < TEST_TOL);
#else
  nomp_test_assert(sum == (N - 1) * N / 2);
#endif

  return 0;
}
#undef nomp_api_500_sum_array

#define nomp_api_500_condition TOKEN_PASTE(nomp_api_500_condition, TEST_SUFFIX)
static int nomp_api_500_condition(int N) {
  nomp_test_assert(N <= 10);

  TEST_TYPE a[10], sum;
  const int mid_point = (int)N / 2;
  for (int i = 0; i < mid_point; ++i)
    a[i] = 0;
  for (int i = mid_point; i < N; ++i)
    a[i] = i;

  const char *knl_fmt =
      "void foo(%s *a, int N, %s *sum) {                               \n"
      "  for (int i = 0; i < N; i++) {                                 \n"
      "    if (a[i] > 0)                                               \n"
      "      sum[0] += 1;                                              \n"
      "  }                                                             \n"
      "}                                                               \n";
  const char *clauses[4] = {"reduce", "sum", "+", NULL};
  nomp_api_500_sum_array_aux(knl_fmt, clauses, a, N, &sum);

#if defined(TEST_TOL)
  nomp_test_assert(fabs(sum - mid_point) < TEST_TOL);
#else
  nomp_test_assert(sum == mid_point);
#endif

  return 0;
}
#undef nomp_api_500_condition
#undef nomp_api_500_sum_array_aux

#define nomp_api_500_prod_aux TOKEN_PASTE(nomp_api_500_prod_aux, TEST_SUFFIX)
static int nomp_api_500_prod_aux(const char *fmt, const char **clauses,
                                 TEST_TYPE *a, TEST_TYPE *b, int n,
                                 TEST_TYPE *c) {
  nomp_test_chk(nomp_update(a, 0, n, sizeof(TEST_TYPE), NOMP_TO));
  nomp_test_chk(nomp_update(b, 0, n, sizeof(TEST_TYPE), NOMP_TO));
  nomp_test_chk(nomp_update(c, 0, n, sizeof(TEST_TYPE), NOMP_TO));

  int id = -1;
  char *knl = generate_knl(fmt, 3, TOSTRING(TEST_TYPE), TOSTRING(TEST_TYPE),
                           TOSTRING(TEST_TYPE));
  nomp_test_chk(nomp_jit(&id, knl, clauses, 4, "a", sizeof(TEST_TYPE), NOMP_PTR,
                         "b", sizeof(TEST_TYPE), NOMP_PTR, "N", sizeof(int),
                         NOMP_INT, "c", sizeof(TEST_TYPE), TEST_NOMP_TYPE));
  nomp_free(&knl);

  nomp_test_chk(nomp_run(id, a, b, &n, c));
  nomp_test_chk(nomp_sync());
  nomp_test_chk(nomp_update(a, 0, n, sizeof(TEST_TYPE), NOMP_FREE));
  nomp_test_chk(nomp_update(b, 0, n, sizeof(TEST_TYPE), NOMP_FREE));
  nomp_test_chk(nomp_update(c, 0, n, sizeof(TEST_TYPE), NOMP_FROM));

  return 0;
}

#define nomp_api_500_mxm TOKEN_PASTE(nomp_api_500_mxm, TEST_SUFFIX)
static int nomp_api_500_mxm(int N) {
  nomp_test_assert(N <= 10);

  TEST_TYPE a[100], b[100], c[100];
  TEST_TYPE output_element = 0;
  for (int i = 0; i < N; i++) {
    output_element += i * i;
    for (int j = 0; j < N; j++)
      a[i * N + j] = b[i + j * N] = j, c[i * N + j] = 0;
  }

  const char *knl_fmt =
      "void foo(%s *a, %s *b, int N, %s *c) {                  \n"
      "  for (int k = 0; k < N; k++)                           \n"
      "    for (int j = 0; j < N; j++)                         \n"
      "      for (int i = 0; i < N; i++)                       \n"
      "        c[i + j * N] += a[i + k * N] * b[k + j * N];    \n"
      "}                                                       \n";

  const char *clauses[4] = {"reduce", "c", "+", NULL};
  nomp_api_500_prod_aux(knl_fmt, clauses, a, b, N, c);

#if defined(TEST_TOL)
  for (unsigned i = 0; i < N * N; i++)
    nomp_test_assert(fabs(c[i] - output_element) < TEST_TOL);
#else
  for (unsigned i = 0; i < N * N; i++)
    nomp_test_assert(c[i] == output_element);
#endif

  return 0;
}
#undef nomp_api_500_mxm

#define nomp_api_500_vxm TOKEN_PASTE(nomp_api_500_vxm, TEST_SUFFIX)
static int nomp_api_500_vxm(int N) {
  nomp_test_assert(N <= 10);

  TEST_TYPE a[100], b[10], c[10];
  TEST_TYPE output_element = 0;
  for (int i = 0; i < N; i++) {
    output_element += i * i;
    b[i] = i, c[i] = 0;
    for (int j = 0; j < N; j++)
      a[i * N + j] = j;
  }

  const char *knl_fmt =
      "void foo(%s *a, %s *b, int N, %s *c) {                  \n"
      "  for (int i = 0; i < N; i++)                           \n"
      "    for (int j = 0; j < N; j++)                         \n"
      "        c[i] += a[j + i * N] * b[j];                    \n"
      "}                                                       \n";

  const char *clauses[4] = {"reduce", "c", "+", NULL};
  nomp_api_500_prod_aux(knl_fmt, clauses, a, b, N, c);

#if defined(TEST_TOL)
  for (unsigned i = 0; i < N; i++)
    nomp_test_assert(fabs(c[i] - output_element) < TEST_TOL);
#else
  for (unsigned i = 0; i < N; i++)
    nomp_test_assert(c[i] == output_element);
#endif

  return 0;
}
#undef nomp_api_500_vxm
#undef nomp_api_500_prod_aux

#define nomp_api_500_dot_aux TOKEN_PASTE(nomp_api_500_dot_aux, TEST_SUFFIX)
static int nomp_api_500_dot_aux(const char *fmt, const char **clauses,
                                TEST_TYPE *a, TEST_TYPE *b, int n,
                                TEST_TYPE *total) {
  nomp_test_chk(nomp_update(a, 0, n, sizeof(TEST_TYPE), NOMP_TO));
  nomp_test_chk(nomp_update(b, 0, n, sizeof(TEST_TYPE), NOMP_TO));

  int id = -1;
  char *knl = generate_knl(fmt, 3, TOSTRING(TEST_TYPE), TOSTRING(TEST_TYPE),
                           TOSTRING(TEST_TYPE));
  nomp_test_chk(nomp_jit(&id, knl, clauses, 4, "a", sizeof(TEST_TYPE *),
                         NOMP_PTR, "b", sizeof(TEST_TYPE), NOMP_PTR, "N",
                         sizeof(int), NOMP_INT, "total", sizeof(TEST_TYPE),
                         TEST_NOMP_TYPE));
  nomp_free(&knl);

  nomp_test_chk(nomp_run(id, a, b, &n, total));
  nomp_test_chk(nomp_sync());
  nomp_test_chk(nomp_update(a, 0, n, sizeof(TEST_TYPE), NOMP_FREE));
  nomp_test_chk(nomp_update(b, 0, n, sizeof(TEST_TYPE), NOMP_FREE));

  return 0;
}

#define nomp_api_500_dot TOKEN_PASTE(nomp_api_500_dot, TEST_SUFFIX)
static int nomp_api_500_dot(int N) {
  nomp_test_assert(N <= 10);

  TEST_TYPE a[10], b[10], total;
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
