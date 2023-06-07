#include "nomp-test.h"

#define TEST_MAX_SIZE2 (TEST_MAX_SIZE * TEST_MAX_SIZE)

#define nomp_api_300_aux TOKEN_PASTE(nomp_api_300_aux, TEST_SUFFIX)
static int nomp_api_300_aux(const char *fmt, TEST_TYPE *a, TEST_TYPE *b,
                            int rows, int cols, int n) {
  nomp_test_chk(nomp_update(a, 0, n, sizeof(TEST_TYPE), NOMP_TO));
  nomp_test_chk(nomp_update(b, 0, n, sizeof(TEST_TYPE), NOMP_TO));

  int id = -1;
  const char *clauses[4] = {"transform", "nomp-api-300", "transform", 0};
  char *knl = generate_knl(fmt, 2, TOSTRING(TEST_TYPE), TOSTRING(TEST_TYPE));
  nomp_test_chk(nomp_jit(&id, knl, clauses, 4, "a", sizeof(TEST_TYPE), NOMP_PTR,
                         "b", sizeof(TEST_TYPE), NOMP_PTR, "rows", sizeof(int),
                         NOMP_INT, "cols", sizeof(int), NOMP_INT));
  nomp_free(&knl);

  nomp_test_chk(nomp_run(id, a, b, &rows, &cols));

  nomp_test_chk(nomp_sync());

  nomp_test_chk(nomp_update(a, 0, n, sizeof(TEST_TYPE), NOMP_FROM));
  nomp_test_chk(nomp_update(a, 0, n, sizeof(TEST_TYPE), NOMP_FREE));
  nomp_test_chk(nomp_update(b, 0, n, sizeof(TEST_TYPE), NOMP_FREE));

  return 0;
}

#define nomp_api_300_add TOKEN_PASTE(nomp_api_300_add, TEST_SUFFIX)
static int nomp_api_300_add(int rows, int cols) {
  const int n = rows * cols;
  nomp_test_assert(n <= TEST_MAX_SIZE2);

  TEST_TYPE a[TEST_MAX_SIZE2], b[TEST_MAX_SIZE2];
  for (unsigned i = 0; i < n; i++)
    a[i] = 2 * n - i, b[i] = i;

  const char *knl_fmt =
      "void foo(%s *a, %s *b, int rows, int cols) {                    \n"
      "  for (int e = 0; e < rows; e++)                                \n"
      "    for (int i = 0; i < cols; i++)                              \n"
      "      a[e * cols + i] = a[e * cols + i] + b[e * cols + i];      \n"
      "}                                                               \n";
  nomp_api_300_aux(knl_fmt, a, b, rows, cols, n);

#if defined(TEST_TOL)
  for (unsigned i = 0; i < n; i++)
    nomp_test_assert(fabs(a[i] - 2 * n) < TEST_TOL);
#else
  for (unsigned i = 0; i < n; i++)
    nomp_test_assert(a[i] == 2 * n);
#endif

  return 0;
}
#undef nomp_api_300_add

#define nomp_api_300_transpose TOKEN_PASTE(nomp_api_300_transpose, TEST_SUFFIX)
static int nomp_api_300_transpose(int rows, int cols) {
  const int n = rows * cols;
  nomp_test_assert(n <= TEST_MAX_SIZE2);

  TEST_TYPE a[TEST_MAX_SIZE2], b[TEST_MAX_SIZE2];
  for (unsigned i = 0; i < rows; i++)
    for (unsigned j = 0; j < cols; j++)
      b[j + i * cols] = (TEST_TYPE)rand();

  const char *knl_fmt =
      "void foo(%s *a, %s *b, int rows, int cols) {                      \n"
      "  for (int j = 0; j < rows; j++)                                  \n"
      "    for (int i = 0; i < cols; i++)                                \n"
      "       a[j + i * rows] = b[i + j * cols];                         \n"
      "}                                                                 \n";
  nomp_api_300_aux(knl_fmt, a, b, rows, cols, n);

#if defined(TEST_TOL)
  for (unsigned i = 0; i < rows; i++)
    for (unsigned j = 0; j < cols; j++)
      nomp_test_assert(fabs(b[j + i * cols] - a[i + j * rows]) < TEST_TOL);
#else
  for (unsigned i = 0; i < rows; i++)
    for (unsigned j = 0; j < cols; j++)
      nomp_test_assert(b[j + i * cols] == a[i + j * rows]);
#endif

  return 0;
}
#undef nomp_api_300_transpose
#undef nomp_api_300_aux

#define nomp_api_300_multiply_aux                                              \
  TOKEN_PASTE(nomp_api_300_multiply_aux, TEST_SUFFIX)
static int nomp_api_300_multiply_aux(const char *fmt, TEST_TYPE *a,
                                     TEST_TYPE *b, TEST_TYPE *c, int n,
                                     int a_size, int b_size,
                                     const char *transform_name) {
  nomp_test_chk(nomp_update(a, 0, a_size, sizeof(TEST_TYPE), NOMP_TO));
  nomp_test_chk(nomp_update(b, 0, b_size, sizeof(TEST_TYPE), NOMP_TO));
  nomp_test_chk(nomp_update(c, 0, b_size, sizeof(TEST_TYPE), NOMP_ALLOC));

  int id = -1;
  const char *clauses[4] = {"transform", "nomp-api-300", transform_name, 0};
  char *knl = generate_knl(fmt, 3, TOSTRING(TEST_TYPE), TOSTRING(TEST_TYPE),
                           TOSTRING(TEST_TYPE));
  nomp_test_chk(nomp_jit(&id, knl, clauses, 4, "a", sizeof(TEST_TYPE), NOMP_PTR,
                         "b", sizeof(TEST_TYPE), NOMP_PTR, "c",
                         sizeof(TEST_TYPE), NOMP_PTR, "size", sizeof(int),
                         NOMP_INT));
  nomp_free(&knl);

  nomp_test_chk(nomp_run(id, a, b, c, &n));

  nomp_test_chk(nomp_sync());

  nomp_test_chk(nomp_update(c, 0, b_size, sizeof(TEST_TYPE), NOMP_FROM));
  nomp_test_chk(nomp_update(a, 0, a_size, sizeof(TEST_TYPE), NOMP_FREE));
  nomp_test_chk(nomp_update(b, 0, b_size, sizeof(TEST_TYPE), NOMP_FREE));
  nomp_test_chk(nomp_update(c, 0, b_size, sizeof(TEST_TYPE), NOMP_FREE));

  return 0;
}

#define nomp_api_300_mxm TOKEN_PASTE(nomp_api_300_mxm, TEST_SUFFIX)
static int nomp_api_300_mxm(int n) {
  nomp_test_assert(n <= TEST_MAX_SIZE2);

  TEST_TYPE a[TEST_MAX_SIZE2], b[TEST_MAX_SIZE2], c[TEST_MAX_SIZE2];
  TEST_TYPE output_element = 0;
  for (int i = 0; i < n; i++) {
    output_element += i * i;
    for (int j = 0; j < n; j++)
      a[i * n + j] = b[i + j * n] = j;
  }

  const char *knl_fmt =
      "void foo(%s *a, %s *b, %s *c, int size) {               \n"
      "  for (unsigned i = 0; i < size; i++) {                 \n"
      "    for (unsigned j = 0; j < size; j++) {               \n"
      "      double dot = 0;                                   \n"
      "      for (unsigned k = 0; k < size; k++)               \n"
      "        dot += a[i * size + k] * b[k * size + j];       \n"
      "      c[i * size + j] = dot;                            \n"
      "    }                                                   \n"
      "  }                                                     \n"
      "}                                                       \n";
  nomp_api_300_multiply_aux(knl_fmt, a, b, c, n, n * n, n * n, "mxm_transform");

#if defined(TEST_TOL)
  for (unsigned i = 0; i < n * n; i++)
    nomp_test_assert(fabs(c[i] - output_element) < TEST_TOL);
#else
  for (unsigned i = 0; i < n * n; i++)
    nomp_test_assert(c[i] == output_element);
#endif

  return 0;
}
#undef nomp_api_300_mxm

#define nomp_api_300_vxm TOKEN_PASTE(nomp_api_300_vxm, TEST_SUFFIX)
static int nomp_api_300_vxm(int n) {
  nomp_test_assert(n <= TEST_MAX_SIZE);

  TEST_TYPE a[TEST_MAX_SIZE2], b[TEST_MAX_SIZE], c[TEST_MAX_SIZE];
  TEST_TYPE output_element = 0;
  for (int i = 0; i < n; i++) {
    output_element += i * i;
    b[i] = i, c[i] = 0;
    for (int j = 0; j < n; j++)
      a[i * n + j] = j;
  }

  const char *knl_fmt =
      "void foo(%s *a, %s *b, %s *c, int size) {               \n"
      "  for (unsigned i = 0; i < size; i++) {                 \n"
      "    double dot = 0;                                     \n"
      "    for (unsigned j = 0; j < size; j++)                 \n"
      "      dot += a[j + i * size] * b[j];                    \n"
      "    c[i] = dot;                                         \n"
      "  }                                                     \n"
      "}                                                       \n";
  nomp_api_300_multiply_aux(knl_fmt, a, b, c, n, n * n, n, "vxm_transform");

#if defined(TEST_TOL)
  for (unsigned i = 0; i < n; i++)
    nomp_test_assert(fabs(c[i] - output_element) < TEST_TOL);
#else
  for (unsigned i = 0; i < n; i++)
    nomp_test_assert(c[i] == output_element);
#endif

  return 0;
}
#undef nomp_api_300_vxm
#undef nomp_api_300_multiply_aux
