#if !defined(_NOMP_LOG_H_)
#define _NOMP_LOG_H_
#include "nomp.h"
#include <stddef.h>

extern const char *ERR_STR_USER_MAP_PTR_IS_INVALID;
extern const char *ERR_STR_USER_DEVICE_IS_INVALID;

extern const char *ERR_STR_RUNTIME_MEMORY_ALLOCATION_FAILURE;

extern const char *ERR_STR_KNL_ARG_TYPE_IS_INVALID;

/**
 * @ingroup nomp_internal_api
 * @brief Register a log with libnomp runtime.
 *
 * @details Register a log given a description of the log, log number and log
 * type. Returns a unique log id that can be used to query log later. Use
 * set_log() macro to py pass the argumnets \p fname and \p line_no.
 *
 * @param[in] logno Log number which is defined in nomp.h
 * @param[in] type Type of the log (one of @ref nomp_log_type)
 * @param[in] desc Detailed description of the log.
 * @param[in] fname File name in which the set_log_() is called.
 * @param[in] line_no Line number where the set_log_() is called.
 * @return int
 */
int set_log_(const char *desc, int logno, nomp_log_type type, const char *fname,
             unsigned line_no, ...);

#define set_log(logno, type, desc, ...)                                        \
  set_log_(desc, logno, type, __FILE__, __LINE__, ##__VA_ARGS__)

#endif
