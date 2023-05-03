#include "nomp-test.h"

#define nomp_api_500_aux TOKEN_PASTE(nomp_api_500_aux, TEST_SUFFIX)
static int nomp_api_500_aux(const char *fmt, const char **clauses, TEST_TYPE *a,
                            int n, TEST_TYPE *sum) {
  nomp_test_chk(nomp_update(a, 0, n, sizeof(TEST_TYPE), NOMP_TO));

  int id = -1;
  char *knl = generate_knl(fmt, 2, TOSTRING(TEST_TYPE), TOSTRING(TEST_TYPE));
  nomp_test_chk(nomp_jit(&id, knl, clauses, 3, "a", sizeof(TEST_TYPE *),
                         NOMP_PTR, "N", sizeof(int), NOMP_INT, "sum",
                         sizeof(TEST_TYPE),
                         TEST_NOMP_TYPE | NOMP_ATTRIBUTE_REDUCTION));
  nomp_free(knl);

  nomp_test_chk(nomp_run(id, a, &n, sum));

  nomp_test_chk(nomp_sync());

  nomp_test_chk(nomp_update(a, 0, n, sizeof(TEST_TYPE), NOMP_FREE));

  return 0;
}

#define nomp_api_500_sum TOKEN_PASTE(nomp_api_500_sum, TEST_SUFFIX)
static int nomp_api_500_sum(int N) {
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
  const char *clauses[1] = {0};
  nomp_api_500_aux(knl_fmt, clauses, a, N, &sum);

#if defined(TEST_TOL)
  nomp_test_assert(fabs(sum - (N - 1) * N / 2) < TEST_TOL);
#else
  nomp_test_assert(sum == (N - 1) * N / 2);
#endif

  return 0;
}
#undef nomp_api_500_sum

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
  const char *clauses[1] = {0};
  nomp_api_500_aux(knl_fmt, clauses, a, N, &sum);

#if defined(TEST_TOL)
  nomp_test_assert(fabs(sum - mid_point) < TEST_TOL);
#else
  nomp_test_assert(sum == mid_point);
#endif

  return 0;
}
#undef nomp_api_500_condition
#undef nomp_api_500_aux

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
  nomp_test_chk(nomp_jit(&id, knl, clauses, 4, "a", sizeof(TEST_TYPE *),
                         NOMP_PTR, "b", sizeof(TEST_TYPE *), NOMP_PTR, "N",
                         sizeof(int), NOMP_INT, "c", sizeof(TEST_TYPE),
                         TEST_NOMP_TYPE | NOMP_ATTRIBUTE_REDUCTION));
  nomp_free(knl);

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
    for (int j = 0; j < N; j++) {
      a[i * N + j] = j;
      b[i + j * N] = j;
      c[i * N + j] = 0;
    }
  }

  const char *knl_fmt =
      "void foo(%s *a, %s *b, int N, %s *c) {                  \n"
      "  for (int k = 0; k < N; k++)                           \n"
      "    for (int j = 0; j < N; j++)                         \n"
      "      for (int i = 0; i < N; i++)                       \n"
      "        c[i + j * N] = a[i + k * N] * b[k + j * N];     \n"
      "}                                                       \n";

  const char *clauses[1] = {0};
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
      "        c[i] = a[j + i * N] * b[j];                     \n"
      "}                                                       \n";

  const char *clauses[1] = {0};
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
#undef nomp_api_500_dot_pro_aux
