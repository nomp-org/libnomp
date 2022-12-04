#include "nomp-test.h"

#define nomp_api_300_aux TOKEN_PASTE(nomp_api_300_aux, TEST_SUFFIX)
int nomp_api_300_aux(TEST_TYPE *a, TEST_TYPE *b, int row, int col) {
  const char *knl_fmt =
      "void foo(%s *a, %s *b, int row, int col) {                        \n"
      "  for (int j = 0; j < row; j++)                                   \n"
      "    for (int i = 0; i < col; i++)                                 \n"
      "       b[j + i * row] = a[i + j * col];                           \n"
      "}                                                                 \n";

  size_t len = strlen(knl_fmt) + 3 * strlen(TOSTRING(TEST_TYPE)) + 1;
  char *knl = tcalloc(char, len);
  snprintf(knl, len, knl_fmt, TOSTRING(TEST_TYPE), TOSTRING(TEST_TYPE));

  static int id = -1;
  const char *clauses[4] = {"transform", "nomp-api-300", "foo", 0};
  int err = nomp_jit(&id, knl, clauses);
  nomp_chk(err);

  err = nomp_run(id, 4, "a", NOMP_PTR, sizeof(TEST_TYPE), a, "b", NOMP_PTR,
                 sizeof(TEST_TYPE), b, "row", NOMP_INT, sizeof(int), &row,
                 "col", NOMP_INT, sizeof(int), &col);
  nomp_chk(err);

  tfree(knl);
  return 0;
}

#define nomp_api_300 TOKEN_PASTE(nomp_api_300, TEST_SUFFIX)
int nomp_api_300(int row, int col) {
  nomp_assert(row <= 10);
  nomp_assert(col <= 10);
  TEST_TYPE a[100], b[100];
  for (unsigned i = 0; i < row; i++)
    for (unsigned j = 0; j < col; j++)
      a[j + i * col] = (TEST_TYPE)rand();

  int err = nomp_update(a, 0, row * col, sizeof(TEST_TYPE), NOMP_TO);
  nomp_chk(err);
  err = nomp_update(b, 0, row * col, sizeof(TEST_TYPE), NOMP_ALLOC);
  nomp_chk(err);

  nomp_api_300_aux(a, b, row, col);

  err = nomp_update(b, 0, row * col, sizeof(TEST_TYPE), NOMP_FROM);
  nomp_chk(err);

#if defined(TEST_TOL)
  for (unsigned i = 0; i < row; i++)
    for (unsigned j = 0; j < col; j++)
      nomp_assert(fabs(a[j + i * col] - b[i + j * row]) < TEST_TOL);
#else
  for (unsigned i = 0; i < row; i++)
    for (unsigned j = 0; j < col; j++)
      nomp_assert(a[j + i * col] == b[i + j * row]);
#endif

  err = nomp_update(a, 0, row * col, sizeof(TEST_TYPE), NOMP_FREE);
  nomp_chk(err);
  err = nomp_update(b, 0, row * col, sizeof(TEST_TYPE), NOMP_FREE);
  nomp_chk(err);

  return 0;
}
#undef nomp_api_300

#undef nomp_api_300_aux
