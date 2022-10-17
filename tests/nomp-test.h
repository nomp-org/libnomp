#if !defined(_NOMP_TEST_H_)
#define _NOMP_TEST_H_

#include <math.h>
#include <stdio.h>
#include <string.h>

#define TOKEN_PASTE_(a, b) a##b
#define TOKEN_PASTE(a, b) TOKEN_PASTE_(a, b)

#define TOSTRING_(x) #x
#define TOSTRING(x) TOSTRING_(x)

#endif // _NOMP_TEST_H_
