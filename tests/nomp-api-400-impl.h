#include "nomp-test.h"

#define nomp_api_400_static_aux                                                \
  TOKEN_PASTE(nomp_api_400_static_aux, TEST_SUFFIX)
static int nomp_api_400_static_aux(const char *fmt, TEST_TYPE *b, TEST_TYPE *a,
                                   int n) {
  char *knl = generate_knl(fmt, 3, TOSTRING(TEST_TYPE), TOSTRING(TEST_TYPE),
                           TOSTRING(TEST_TYPE));

  int id = -1;
  const char *clauses[4] = {"transform", "nomp_api_400", "transform", 0};
  nomp_test_check(nomp_jit(&id, knl, clauses, 3, "b", sizeof(TEST_TYPE),
                           NOMP_PTR, "a", sizeof(TEST_TYPE), NOMP_PTR, "n",
                           sizeof(int), NOMP_INT));
  nomp_free(&knl);

  nomp_test_check(nomp_update(a, 0, TEST_MAX_SIZE, sizeof(TEST_TYPE), NOMP_TO));
  nomp_test_check(nomp_update(b, 0, TEST_MAX_SIZE, sizeof(TEST_TYPE), NOMP_TO));

  nomp_test_check(nomp_run(id, b, a, &n));

  nomp_test_check(
      nomp_update(b, 0, TEST_MAX_SIZE, sizeof(TEST_TYPE), NOMP_FROM));
  nomp_test_check(
      nomp_update(b, 0, TEST_MAX_SIZE, sizeof(TEST_TYPE), NOMP_FREE));
  nomp_test_check(
      nomp_update(a, 0, TEST_MAX_SIZE, sizeof(TEST_TYPE), NOMP_FREE));

  return 0;
}

#define nomp_api_400_fixed_size_1d_array                                       \
  TOKEN_PASTE(nomp_api_400_fixed_size_1d_array, TEST_SUFFIX)
static int nomp_api_400_fixed_size_1d_array(int n) {
  nomp_test_assert(n <= TEST_MAX_SIZE / 32);

  const char *knl_fmt =
      "void foo(%s *b, const %s *a, int n) {                           \n"
      "  for (int i = 0; i < n; i++) {                                 \n"
      "    %s s[32];                                                   \n"
      "    for (int j = 0; j < 32; j++)                                \n"
      "      s[j] = a[i * 32 + j];                                     \n"
      "    for (int j = 0; j < 32; j++)                                \n"
      "      s[j] += s[j];                                             \n"
      "    for (int j = 0; j < 32; j++)                                \n"
      "      b[i * 32 + j] = s[j];                                     \n"
      "  }                                                             \n"
      "}                                                               \n";

  TEST_TYPE a[TEST_MAX_SIZE], b[TEST_MAX_SIZE];
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 32; j++)
      a[i * 32 + j] = i;
  }

  nomp_api_400_static_aux(knl_fmt, b, a, n);

#if defined(TEST_TOL)
  for (int i = 0; i < 32 * n; i++)
    nomp_test_assert(fabs(b[i] - (TEST_TYPE)(2 * a[i])) < TEST_TOL);
#else
  for (int i = 0; i < 32 * n; i++)
    nomp_test_assert(b[i] == (TEST_TYPE)(2 * a[i]));
#endif

  return 0;
}
#undef nomp_api_400_fixed_size_1d_array

#define nomp_api_400_dynamic_aux                                               \
  TOKEN_PASTE(nomp_api_400_dynamic_aux, TEST_SUFFIX)
static int nomp_api_400_dynamic_aux(const char *fmt, TEST_TYPE *b, TEST_TYPE *a,
                                    int n, int m) {
  char *knl = generate_knl(fmt, 3, TOSTRING(TEST_TYPE), TOSTRING(TEST_TYPE),
                           TOSTRING(TEST_TYPE));

  int id = -1;
  const char *clauses[4] = {"transform", "nomp_api_400", "transform", 0};
  nomp_test_check(nomp_jit(&id, knl, clauses, 4, "b", sizeof(TEST_TYPE),
                           NOMP_PTR, "a", sizeof(TEST_TYPE), NOMP_PTR, "n",
                           sizeof(int), NOMP_INT, "m", sizeof(int),
                           NOMP_INT | NOMP_JIT, &m));
  nomp_free(&knl);

  nomp_test_check(nomp_update(a, 0, TEST_MAX_SIZE, sizeof(TEST_TYPE), NOMP_TO));
  nomp_test_check(nomp_update(b, 0, TEST_MAX_SIZE, sizeof(TEST_TYPE), NOMP_TO));

  nomp_test_check(nomp_run(id, b, a, &n));

  nomp_test_check(
      nomp_update(b, 0, TEST_MAX_SIZE, sizeof(TEST_TYPE), NOMP_FROM));
  nomp_test_check(
      nomp_update(b, 0, TEST_MAX_SIZE, sizeof(TEST_TYPE), NOMP_FREE));
  nomp_test_check(
      nomp_update(a, 0, TEST_MAX_SIZE, sizeof(TEST_TYPE), NOMP_FREE));

  return 0;
}

#define nomp_api_400_dynamic_1d_array                                          \
  TOKEN_PASTE(nomp_api_400_dynamic_1d_array, TEST_SUFFIX)
static int nomp_api_400_dynamic_1d_array(int n, int m) {
  nomp_test_assert(n * m <= TEST_MAX_SIZE);

  const char *knl_fmt =
      "void foo(%s *b, const %s *a, int n, int m) {                    \n"
      "  for (int i = 0; i < n; i++) {                                 \n"
      "    %s s[m];                                                    \n"
      "    for (int j = 0; j < m; j++)                                 \n"
      "      s[j] = a[i * m + j];                                      \n"
      "    for (int j = 0; j < m; j++)                                 \n"
      "      s[j] += s[j];                                             \n"
      "    for (int j = 0; j < m; j++)                                 \n"
      "      b[i * m + j] = s[j];                                      \n"
      "  }                                                             \n"
      "}                                                               \n";

  TEST_TYPE a[TEST_MAX_SIZE], b[TEST_MAX_SIZE];
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++)
      a[i * m + j] = i;
  }

  nomp_api_400_dynamic_aux(knl_fmt, b, a, n, m);

#if defined(TEST_TOL)
  for (int i = 0; i < n * m; i++)
    nomp_test_assert(fabs(b[i] - (TEST_TYPE)(2 * a[i])) < TEST_TOL);
#else
  for (int i = 0; i < n * m; i++)
    nomp_test_assert(b[i] == (TEST_TYPE)(2 * a[i]));
#endif

  return 0;
}
#undef nomp_api_400_dynamic_1d_array

#define nomp_api_400_fixed_size_2d_array                                       \
  TOKEN_PASTE(nomp_api_400_fixed_size_2d_array, TEST_SUFFIX)
static int nomp_api_400_fixed_size_2d_array(int n) {
  nomp_test_assert(n <= TEST_MAX_SIZE / 32);

  const char *knl_fmt =
      "void foo(%s *b, const %s *a, int n) {                           \n"
      "  for (int i = 0; i < n; i++) {                                 \n"
      "    %s s[32][8];                                                \n"
      "    for (int j = 0; j < 32; j++)                                \n"
      "      s[j][4] = a[i * 32 + j];                                  \n"
      "    for (int j = 0; j < 32; j++)                                \n"
      "      s[j][4] += s[j][4];                                       \n"
      "    for (int j = 0; j < 32; j++)                                \n"
      "      b[i * 32 + j] = s[j][4];                                  \n"
      "  }                                                             \n"
      "}                                                               \n";

  TEST_TYPE a[TEST_MAX_SIZE], b[TEST_MAX_SIZE];
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 32; j++)
      a[i * 32 + j] = i;
  }

  nomp_api_400_static_aux(knl_fmt, b, a, n);

#if defined(TEST_TOL)
  for (int i = 0; i < 32 * n; i++)
    nomp_test_assert(fabs(b[i] - (TEST_TYPE)(2 * a[i])) < TEST_TOL);
#else
  for (int i = 0; i < 32 * n; i++)
    nomp_test_assert(b[i] == (TEST_TYPE)(2 * a[i]));
#endif

  return 0;
}
#undef nomp_api_400_fixed_size_2d_array

#define nomp_api_400_dynamic_2d_array                                          \
  TOKEN_PASTE(nomp_api_400_dynamic_2d_array, TEST_SUFFIX)
static int nomp_api_400_dynamic_2d_array(int n, int m) {
  nomp_test_assert(n * m <= TEST_MAX_SIZE);
  nomp_test_assert(m > 4);

  const char *knl_fmt =
      "void foo(%s *b, const %s *a, int n, int m) {                    \n"
      "  for (int i = 0; i < n; i++) {                                 \n"
      "    %s s[m][m];                                                 \n"
      "    for (int j = 0; j < m; j++)                                 \n"
      "      s[j][4] = a[i * m + j];                                   \n"
      "    for (int j = 0; j < m; j++)                                 \n"
      "      s[j][4] += s[j][4];                                       \n"
      "    for (int j = 0; j < m; j++)                                 \n"
      "      b[i * m + j] = s[j][4];                                   \n"
      "  }                                                             \n"
      "}                                                               \n";

  TEST_TYPE a[TEST_MAX_SIZE], b[TEST_MAX_SIZE];
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++)
      a[i * m + j] = i;
  }

  nomp_api_400_dynamic_aux(knl_fmt, b, a, n, m);

#if defined(TEST_TOL)
  for (int i = 0; i < n * m; i++)
    nomp_test_assert(fabs(b[i] - (TEST_TYPE)(2 * a[i])) < TEST_TOL);
#else
  for (int i = 0; i < n * m; i++)
    nomp_test_assert(b[i] == (TEST_TYPE)(2 * a[i]));
#endif

  return 0;
}
#undef nomp_api_400_dynamic_2d_array

#define nomp_api_400_fixed_size_3d_array                                       \
  TOKEN_PASTE(nomp_api_400_fixed_size_3d_array, TEST_SUFFIX)
static int nomp_api_400_fixed_size_3d_array(int n) {
  nomp_test_assert(n <= TEST_MAX_SIZE / 32);

  const char *knl_fmt =
      "void foo(%s *b, const %s *a, int n) {                           \n"
      "  for (int i = 0; i < n; i++) {                                 \n"
      "    %s s[2][2][32];                                             \n"
      "    for (int j = 0; j < 32; j++)                                \n"
      "      s[1][1][j] = a[i * 32 + j];                               \n"
      "    for (int j = 0; j < 32; j++)                                \n"
      "      s[1][1][j] += s[1][1][j];                                 \n"
      "    for (int j = 0; j < 32; j++)                                \n"
      "      b[i * 32 + j] = s[1][1][j];                               \n"
      "  }                                                             \n"
      "}                                                               \n";

  TEST_TYPE a[TEST_MAX_SIZE], b[TEST_MAX_SIZE];
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 32; j++)
      a[i * 32 + j] = i;
  }

  nomp_api_400_static_aux(knl_fmt, b, a, n);

#if defined(TEST_TOL)
  for (int i = 0; i < 32 * n; i++)
    nomp_test_assert(fabs(b[i] - (TEST_TYPE)(2 * a[i])) < TEST_TOL);
#else
  for (int i = 0; i < 32 * n; i++)
    nomp_test_assert(b[i] == (TEST_TYPE)(2 * a[i]));
#endif

  return 0;
}
#undef nomp_api_400_fixed_size_3d_array
#undef nomp_api_400_static_aux

#define nomp_api_400_dynamic_3d_array                                          \
  TOKEN_PASTE(nomp_api_400_dynamic_3d_array, TEST_SUFFIX)
static int nomp_api_400_dynamic_3d_array(int n, int m) {
  nomp_test_assert(n * m <= TEST_MAX_SIZE);

  const char *knl_fmt =
      "void foo(%s *b, const %s *a, int n, int m) {                    \n"
      "  for (int i = 0; i < n; i++) {                                 \n"
      "    %s s[m][m][m];                                              \n"
      "    for (int j = 0; j < m; j++)                                 \n"
      "      s[j][0][0] = a[i * m + j];                                \n"
      "    for (int j = 0; j < m; j++)                                 \n"
      "      s[j][0][0] += s[j][0][0];                                 \n"
      "    for (int j = 0; j < m; j++)                                 \n"
      "      b[i * m + j] = s[j][0][0];                                \n"
      "  }                                                             \n"
      "}                                                               \n";

  TEST_TYPE a[TEST_MAX_SIZE], b[TEST_MAX_SIZE];
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++)
      a[i * m + j] = i;
  }

  nomp_api_400_dynamic_aux(knl_fmt, b, a, n, m);

#if defined(TEST_TOL)
  for (int i = 0; i < n * m; i++)
    nomp_test_assert(fabs(b[i] - (TEST_TYPE)(2 * a[i])) < TEST_TOL);
#else
  for (int i = 0; i < n * m; i++)
    nomp_test_assert(b[i] == (TEST_TYPE)(2 * a[i]));
#endif

  return 0;
}
#undef nomp_api_400_dynamic_3d_array
#undef nomp_api_400_dynamic_aux
