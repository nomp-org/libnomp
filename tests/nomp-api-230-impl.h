#include "nomp-test.h"

#define nomp_api_230_aux TOKEN_PASTE(nomp_api_230_aux, TEST_SUFFIX)
static int nomp_api_230_aux(const char *knl_fmt, const char **clauses,
                            TEST_TYPE *a, TEST_TYPE *b, int rows, int cols,
                            int n) {
  int err = nomp_update(a, 0, n, sizeof(TEST_TYPE), NOMP_TO);
  nomp_test_chk(err);
  err = nomp_update(b, 0, n, sizeof(TEST_TYPE), NOMP_TO);
  nomp_test_chk(err);

  int id = -1;
  err = create_knl(&id, knl_fmt, clauses, 2, TOSTRING(TEST_TYPE),
                   TOSTRING(TEST_TYPE));
  nomp_test_chk(err);
  nomp_run(id, 4, "a", NOMP_PTR, sizeof(TEST_TYPE), a, "b", NOMP_PTR,
           sizeof(TEST_TYPE), b, "rows", NOMP_INTEGER, sizeof(int), &rows,
           "cols", NOMP_INTEGER, sizeof(int), &cols);
  nomp_test_chk(err);

  err = nomp_update(a, 0, n, sizeof(TEST_TYPE), NOMP_FROM);
  nomp_test_chk(err);
  err = nomp_update(a, 0, n, sizeof(TEST_TYPE), NOMP_FREE);
  nomp_test_chk(err);
  err = nomp_update(b, 0, n, sizeof(TEST_TYPE), NOMP_FREE);
  nomp_test_chk(err);

  return 0;
}

#define nomp_api_230_add TOKEN_PASTE(nomp_api_230_add, TEST_SUFFIX)
static int nomp_api_230_add(int rows, int cols) {
  const int n = rows * cols;
  nomp_test_assert(n <= 256);

  TEST_TYPE a[256], b[256];
  for (unsigned i = 0; i < n; i++)
    a[i] = 2 * n - i, b[i] = i;

  const char *knl_fmt =
      "void foo(%s *a, %s *b, int rows, int cols) {                    \n"
      "  for (int e = 0; e < rows; e++)                                \n"
      "    for (int i = 0; i < cols; i++)                              \n"
      "      a[e * cols + i] = a[e * cols + i] + b[e * cols + i];      \n"
      "}                                                               \n";
  const char *clauses[7] = {"annotate",     "dof_loop", "i", "annotate",
                            "element_loop", "e",        0};
  nomp_api_230_aux(knl_fmt, clauses, a, b, rows, cols, n);

#if defined(TEST_TOL)
  for (unsigned i = 0; i < n; i++)
    nomp_test_assert(fabs(a[i] - 2 * n) < TEST_TOL);
#else
  for (unsigned i = 0; i < n; i++)
    nomp_test_assert(a[i] == 2 * n);
#endif

  return 0;
}
#undef nomp_api_230_add

#define nomp_api_230_transform TOKEN_PASTE(nomp_api_230_transform, TEST_SUFFIX)
static int nomp_api_230_transform(int rows, int cols) {
  const int n = rows * cols;
  TEST_TYPE a[n], b[n];
  for (unsigned i = 0; i < rows; i++)
    for (unsigned j = 0; j < cols; j++)
      b[j + i * cols] = (TEST_TYPE)rand();

  const char *knl_fmt =
      "void foo(%s *a, %s *b, int rows, int cols) {                      \n"
      "  for (int j = 0; j < rows; j++)                                  \n"
      "    for (int i = 0; i < cols; i++)                                \n"
      "       a[j + i * rows] = b[i + j * cols];                         \n"
      "}                                                                 \n";
  const char *clauses[4] = {"transform", "nomp-api-200", "matrix_transform", 0};
  nomp_api_230_aux(knl_fmt, clauses, a, b, rows, cols, n);

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
#undef nomp_api_230_transform
#undef nomp_api_230_aux
