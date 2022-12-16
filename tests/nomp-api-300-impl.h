#include "nomp-test.h"

#define nomp_api_300_aux TOKEN_PASTE(nomp_api_300_aux, TEST_SUFFIX)
int nomp_api_300_aux(TEST_TYPE *a, TEST_TYPE *b, int row, int col) {
  const char *knl_fmt =
      "void foo(%s *a, %s *b, int row, int col) {                        \n"
      "  for (int j = 0; j < row; j++)                                   \n"
      "    for (int i = 0; i < col; i++)                                 \n"
      "       b[j + i * row] = a[i + j * col];                           \n"
      "}                                                                 \n";

  const char *clauses[4] = {"transform", "nomp-api-300", "transform", 0};

  char *knl = create_knl(knl_fmt, 2, TOSTRING(TEST_TYPE), TOSTRING(TEST_TYPE));
  return run_kernel(knl, clauses, 4, "a", NOMP_PTR, sizeof(TEST_TYPE), a, "b",
                    NOMP_PTR, sizeof(TEST_TYPE), b, "row", NOMP_INTEGER,
                    sizeof(int), &row, "col", NOMP_INTEGER, sizeof(int), &col);
}

#define nomp_api_300 TOKEN_PASTE(nomp_api_300, TEST_SUFFIX)
int nomp_api_300(int row, int col) {
  nomp_test_assert(row <= 10);
  nomp_test_assert(col <= 10);
  TEST_TYPE a[100], b[100];
  for (unsigned i = 0; i < row; i++)
    for (unsigned j = 0; j < col; j++)
      a[j + i * col] = (TEST_TYPE)rand();

  int err = nomp_update(a, 0, row * col, sizeof(TEST_TYPE), NOMP_TO);
  nomp_test_chk(err);
  err = nomp_update(b, 0, row * col, sizeof(TEST_TYPE), NOMP_ALLOC);
  nomp_test_chk(err);

  nomp_api_300_aux(a, b, row, col);

  err = nomp_update(b, 0, row * col, sizeof(TEST_TYPE), NOMP_FROM);
  nomp_test_chk(err);

#if defined(TEST_TOL)
  for (unsigned i = 0; i < row; i++)
    for (unsigned j = 0; j < col; j++)
      nomp_test_assert(fabs(a[j + i * col] - b[i + j * row]) < TEST_TOL);
#else
  for (unsigned i = 0; i < row; i++)
    for (unsigned j = 0; j < col; j++)
      nomp_test_assert(a[j + i * col] == b[i + j * row]);
#endif

  err = nomp_update(a, 0, row * col, sizeof(TEST_TYPE), NOMP_FREE);
  nomp_test_chk(err);
  err = nomp_update(b, 0, row * col, sizeof(TEST_TYPE), NOMP_FREE);
  nomp_test_chk(err);

  return 0;
}
#undef nomp_api_300

#undef nomp_api_300_aux
