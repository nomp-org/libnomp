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
      "void foo(%s *a,  int N, %s *sum) {                              \n"
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

  TEST_TYPE a[N], sum;
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
