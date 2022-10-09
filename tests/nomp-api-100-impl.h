#include "nomp-test.h"

#if defined(TEST_TOL)
#include <math.h>
#endif

#define nomp_api_100 TOKEN_PASTE(nomp_api_100, TEST_SUFFIX)
int nomp_api_100(unsigned s, unsigned e) {
  nomp_assert(e <= 10);

  TEST_TYPE a[10] = {0}, b[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

  // Free'ing before mapping should return an error
  int err = nomp_update(a, s, e, sizeof(TEST_TYPE), NOMP_FREE);
  nomp_assert(err == NOMP_INVALID_MAP_PTR);

  // D2H before H2D should return an error
  err = nomp_update(a, s, e, sizeof(TEST_TYPE), NOMP_FROM);
  nomp_assert(err == NOMP_INVALID_MAP_PTR);

  // Mapping H2D multiple times is not an error
  err = nomp_update(b, s, e, sizeof(TEST_TYPE), NOMP_TO);
  nomp_chk(err);
  err = nomp_update(b, s, e, sizeof(TEST_TYPE), NOMP_TO);
  nomp_chk(err);

  err = nomp_update(a, s, e, sizeof(TEST_TYPE), NOMP_TO);
  nomp_chk(err);

  // Mapping D2H multiple times is not an error
  err = nomp_update(a, s, e, sizeof(TEST_TYPE), NOMP_FROM);
  nomp_chk(err);
  err = nomp_update(a, s, e, sizeof(TEST_TYPE), NOMP_FROM);
  nomp_chk(err);

  // Set b to all 0's, then copy the value on the device and
  // check if it is all 1's
  for (unsigned i = 0; i < e; i++)
    b[i] = 0;

  err = nomp_update(b, s, e, sizeof(TEST_TYPE), NOMP_FROM);
  nomp_chk(err);
#if defined(TEST_TOL)
  for (unsigned i = 0; i < s; i++)
    nomp_assert(fabs(b[i]) < TEST_TOL);
  for (unsigned i = s; i < e; i++)
    nomp_assert(fabs(b[i] - 1) < TEST_TOL);
#else
  for (unsigned i = 0; i < s; i++)
    nomp_assert(b[i] == 0);
  for (unsigned i = s; i < e; i++)
    nomp_assert(b[i] == 1);
#endif

  // Free'ing after mapping is not an error
  err = nomp_update(a, s, e, sizeof(TEST_TYPE), NOMP_FREE);
  nomp_chk(err);
  err = nomp_update(b, s, e, sizeof(TEST_TYPE), NOMP_FREE);
  nomp_chk(err);

  return 0;
}
#undef nomp_api_100
