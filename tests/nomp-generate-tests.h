#define TEST_NOMP_TYPE NOMP_INT
#define TEST_TYPE      int
#define TEST_SUFFIX    _int
#include TEST_IMPL_H
#undef TEST_SUFFIX
#undef TEST_TYPE

#define TEST_TYPE   long
#define TEST_SUFFIX _long
#include TEST_IMPL_H
#undef TEST_SUFFIX
#undef TEST_TYPE
#undef TEST_NOMP_TYPE

#define TEST_NOMP_TYPE NOMP_UINT
#define TEST_TYPE      unsigned
#define TEST_SUFFIX    _unsigned
#include TEST_IMPL_H
#undef TEST_SUFFIX
#undef TEST_TYPE

#define TEST_TYPE   unsigned long
#define TEST_SUFFIX _unsigned_long
#include TEST_IMPL_H
#undef TEST_SUFFIX
#undef TEST_TYPE
#undef TEST_NOMP_TYPE

#ifndef TEST_INT_ONLY

#define TEST_NOMP_TYPE NOMP_FLOAT
#define TEST_TOL       1e-12
#define TEST_TYPE      double
#define TEST_SUFFIX    _double
#include TEST_IMPL_H
#undef TEST_SUFFIX
#undef TEST_TYPE
#undef TEST_TOL

#define TEST_TOL    1e-8
#define TEST_TYPE   float
#define TEST_SUFFIX _float
#include TEST_IMPL_H
#undef TEST_SUFFIX
#undef TEST_TYPE
#undef TEST_TOL
#undef TEST_NOMP_TYPE

#endif // TEST_INT_ONLY
