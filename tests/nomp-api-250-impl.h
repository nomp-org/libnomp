#include "nomp-test.h"

#define nomp_api_250_aux TOKEN_PASTE(nomp_api_250_aux, TEST_SUFFIX)
static int nomp_api_250_aux(const char *knl_fmt, TEST_TYPE *a, int n) {
  int err = nomp_update(a, 0, n, sizeof(TEST_TYPE), NOMP_TO);
  nomp_test_chk(err);

  const char *clauses[4] = {"transform", "nomp-api-250", "transform", 0};
  int id = -1;
  err = create_knl(&id, knl_fmt, clauses, 1, TOSTRING(TEST_TYPE),
                   TOSTRING(TEST_TYPE));
  nomp_test_chk(err);

  err = nomp_run(id, 2, "a", NOMP_PTR, sizeof(TEST_TYPE), a, "N", NOMP_INT,
                 sizeof(int), &n);
  nomp_test_chk(err);

  nomp_test_chk(nomp_sync());

  err = nomp_update(a, 0, n, sizeof(TEST_TYPE), NOMP_FROM);
  nomp_test_chk(err);
  err = nomp_update(a, 0, n, sizeof(TEST_TYPE), NOMP_FREE);
  nomp_test_chk(err);

  return 0;
}

#define nomp_api_250_bitwise_and_op                                            \
  TOKEN_PASTE(nomp_api_250_bitwise_and_op, TEST_SUFFIX)
static int nomp_api_250_bitwise_and_op(int n) {
  nomp_test_assert(n <= 10);

  TEST_TYPE a[20];

  const char *knl_fmt =
      "void foo(%s *a, int N) {                                        \n"
      "  for (int i = 0; i < N; i++)                                   \n"
      "    a[i] = i & 3;                                               \n"
      "}                                                               \n";
  nomp_api_250_aux(knl_fmt, a, n);

#if defined(TEST_TOL)
  for (unsigned i = 0; i < n; i++)
    nomp_test_assert(fabs(a[i] - (i & 3)) < TEST_TOL);
#else
  for (unsigned i = 0; i < n; i++)
    nomp_test_assert(a[i] == (i & 3));
#endif

  return 0;
}
#undef nomp_api_250_bitwise_and_op

#define nomp_api_250_bitwise_or_op                                             \
  TOKEN_PASTE(nomp_api_250_bitwise_or_op, TEST_SUFFIX)
static int nomp_api_250_bitwise_or_op(int n) {
  nomp_test_assert(n <= 10);

  TEST_TYPE a[20];

  const char *knl_fmt =
      "void foo(%s *a, int N) {                                        \n"
      "  for (int i = 0; i < N; i++)                                   \n"
      "    a[i] = i | 3;                                               \n"
      "}                                                               \n";
  nomp_api_250_aux(knl_fmt, a, n);

#if defined(TEST_TOL)
  for (unsigned i = 0; i < n; i++)
    nomp_test_assert(fabs(a[i] - (i | 3)) < TEST_TOL);
#else
  for (unsigned i = 0; i < n; i++)
    nomp_test_assert(a[i] == (i | 3));
#endif

  return 0;
}
#undef nomp_api_250_bitwise_or_op

#define nomp_api_250_bitwise_xor_op                                            \
  TOKEN_PASTE(nomp_api_250_bitwise_xor_op, TEST_SUFFIX)
static int nomp_api_250_bitwise_xor_op(int n) {
  nomp_test_assert(n <= 10);

  TEST_TYPE a[20];

  const char *knl_fmt =
      "void foo(%s *a, int N) {                                        \n"
      "  for (int i = 0; i < N; i++)                                   \n"
      "    a[i] = i ^ 3;                                               \n"
      "}                                                               \n";
  nomp_api_250_aux(knl_fmt, a, n);

#if defined(TEST_TOL)
  for (unsigned i = 0; i < n; i++)
    nomp_test_assert(fabs(a[i] - (i ^ 3)) < TEST_TOL);
#else
  for (unsigned i = 0; i < n; i++)
    nomp_test_assert(a[i] == (i ^ 3));
#endif

  return 0;
}
#undef nomp_api_250_bitwise_xor_op

#define nomp_api_250_bitwise_left_shift_op                                     \
  TOKEN_PASTE(nomp_api_250_bitwise_left_shift_op, TEST_SUFFIX)
static int nomp_api_250_bitwise_left_shift_op(int n) {
  nomp_test_assert(n <= 10);

  TEST_TYPE a[20];

  const char *knl_fmt =
      "void foo(%s *a, int N) {                                        \n"
      "  for (int i = 0; i < N; i++)                                   \n"
      "    a[i] = i << 3;                                              \n"
      "}                                                               \n";
  nomp_api_250_aux(knl_fmt, a, n);

#if defined(TEST_TOL)
  for (unsigned i = 0; i < n; i++)
    nomp_test_assert(fabs(a[i] - (i << 3)) < TEST_TOL);
#else
  for (unsigned i = 0; i < n; i++)
    nomp_test_assert(a[i] == (i << 3));
#endif

  return 0;
}
#undef nomp_api_250_bitwise_left_shift_op

#define nomp_api_250_bitwise_right_shift_op                                    \
  TOKEN_PASTE(nomp_api_250_bitwise_right_shift_op, TEST_SUFFIX)
static int nomp_api_250_bitwise_right_shift_op(int n) {
  nomp_test_assert(n <= 10);

  TEST_TYPE a[20];

  const char *knl_fmt =
      "void foo(%s *a, int N) {                                        \n"
      "  for (int i = 0; i < N; i++)                                   \n"
      "    a[i] = i >> 3;                                              \n"
      "}                                                               \n";
  nomp_api_250_aux(knl_fmt, a, n);

#if defined(TEST_TOL)
  for (unsigned i = 0; i < n; i++)
    nomp_test_assert(fabs(a[i] - (i >> 3)) < TEST_TOL);
#else
  for (unsigned i = 0; i < n; i++)
    nomp_test_assert(a[i] == (i >> 3));
#endif

  return 0;
}
#undef nomp_api_250_bitwise_right_shift_op

#define nomp_api_250_bitwise_complement_op                                     \
  TOKEN_PASTE(nomp_api_250_bitwise_complement_op, TEST_SUFFIX)
static int nomp_api_250_bitwise_complement_op(int n) {
  nomp_test_assert(n <= 10);

  TEST_TYPE a[20];
  for (unsigned i = 0; i < n; i++)
    a[i] = 0;

  const char *knl_fmt =
      "void foo(%s *a, int N) {                                        \n"
      "  for (int i = 0; i < N; i++)                                   \n"
      "    a[i] = ~ i;                                                 \n"
      "}                                                               \n";
  nomp_api_250_aux(knl_fmt, a, n);

#if defined(TEST_TOL)
  for (int i = 0; i < n; i++)
    nomp_test_assert(fabs(a[i] - (~i)) < TEST_TOL);
#else
  for (int i = 0; i < n; i++)
    nomp_test_assert(a[i] == (~i));
#endif

  return 0;
}
#undef nomp_api_250_bitwise_complement_op
#undef nomp_api_250_aux
