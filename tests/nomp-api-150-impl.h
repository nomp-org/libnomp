#include "nomp-test.h"

// Calling nomp_run() with invalid kernel_id should return an error.
#define nomp_api_150_invalid_kernel_id                                         \
  TOKEN_PASTE(nomp_api_150_invalid_kernel_id, TEST_SUFFIX)
static int nomp_api_150_invalid_kernel_id(int n) {
  assert(n <= TEST_MAX_SIZE);

  const char *fmt = "void foo(%s *a, %s *b, int N) {                        \n"
                    "  for (int i = 0; i < N; i++)                          \n"
                    "    a[i] = a[i] * b[i];                                \n"
                    "}                                                      \n";

  char *knl = generate_knl(fmt, 2, TOSTRING(TEST_TYPE), TOSTRING(TEST_TYPE));

  int         id         = -1;
  const char *clauses[4] = {"transform", "nomp_api_100", "tile", 0};
  nomp_test_check(nomp_jit(&id, knl, clauses, 3, "a", sizeof(TEST_TYPE),
                           NOMP_PTR, "b", sizeof(TEST_TYPE), NOMP_PTR, "N",
                           sizeof(int), NOMP_INT));
  nomp_free(&knl);

  TEST_TYPE a[TEST_MAX_SIZE], b[TEST_MAX_SIZE];
  int       err = nomp_run(-1, a, b, &n);
  nomp_test_assert(nomp_get_err_no(err) == NOMP_USER_INPUT_IS_INVALID);

  char *log = nomp_get_err_str(err);
  int   eq  = logcmp(log, "\\[Error\\] .*\\/src\\/nomp.[c|cpp]:[0-9]* Kernel "
                             "id -1 passed to nomp_run is not valid.");
  nomp_free(&log);
  nomp_test_assert(eq);

  return 0;
}
#undef nomp_api_150_invalid_kernel_id

// Calling nomp_run() should fail because `b` is not mapped.
#define nomp_api_150_unmapped_array                                            \
  TOKEN_PASTE(nomp_api_150_unmapped_array, TEST_SUFFIX)
static int nomp_api_150_unmapped_array(int n) {
  assert(n <= TEST_MAX_SIZE);

  const char *fmt = "void foo(%s *a, %s *b, int N) {                        \n"
                    "  for (int i = 0; i < N; i++)                          \n"
                    "    a[i] = a[i] * b[i];                                \n"
                    "}                                                      \n";

  char *knl = generate_knl(fmt, 2, TOSTRING(TEST_TYPE), TOSTRING(TEST_TYPE));

  int         id         = -1;
  const char *clauses[4] = {"transform", "nomp_api_100", "tile", 0};
  nomp_test_check(nomp_jit(&id, knl, clauses, 3, "a", sizeof(TEST_TYPE),
                           NOMP_PTR, "b", sizeof(TEST_TYPE), NOMP_PTR, "N",
                           sizeof(int), NOMP_INT));
  nomp_free(&knl);

  TEST_TYPE a[TEST_MAX_SIZE], b[TEST_MAX_SIZE];
  nomp_test_check(nomp_update(a, 0, n, sizeof(TEST_TYPE), NOMP_TO));

  int err = nomp_run(id, a, b, &n);
  nomp_test_assert(nomp_get_err_no(err) == NOMP_USER_MAP_PTR_IS_INVALID);

  char *desc = nomp_get_err_str(err);
  int eq = logcmp(desc, "\\[Error\\] .*\\/src\\/.*.[c|cpp]:[0-9]* Map pointer "
                        "0[xX][0-9a-fA-F]* was not found on device.");
  nomp_free(&desc);
  nomp_test_assert(eq);

  nomp_test_check(nomp_update(a, 0, n, sizeof(TEST_TYPE), NOMP_FREE));

  return 0;
}
#undef nomp_api_150_unmapped_array
