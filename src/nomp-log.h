#if !defined(_NOMP_LOG_H_)
#define _NOMP_LOG_H_

#include <stddef.h>

#include "nomp.h"

extern const char *ERR_STR_USER_MAP_PTR_IS_INVALID;
extern const char *ERR_STR_USER_DEVICE_IS_INVALID;

/**
 * @defgroup nomp_log_utils Internal functions used by logging module.
 */

/**
 * @ingroup nomp_log_utils
 * @brief Register a log with libnomp runtime.
 *
 * @details Register a log given a description of the log, log number and log
 * type. Returns a unique log id that can be used to query log later on success.
 * On failure, nomp_set_log_() returns -1. Use nomp_set_log() macro to by pass
 * the argumnets \p fname and \p line_no.
 *
 * @param[in] logno Log number which is defined in nomp.h
 * @param[in] type Type of the log (one of @ref nomp_log_type)
 * @param[in] desc Detailed description of the log.
 * @param[in] fname File name in which the nomp_set_log_() is called.
 * @param[in] line_no Line number where the nomp_set_log_() is called.
 * @return int
 */
int nomp_set_log_(const char *desc, int logno, nomp_log_type type,
                  const char *fname, unsigned line_no, ...);

#define nomp_set_log(logno, type, desc, ...)                                   \
  nomp_set_log_(desc, logno, type, __FILE__, __LINE__, ##__VA_ARGS__)

/**
 * @ingroup nomp_log_utils
 * @brief Free log variables.
 *
 * @return void
 */
void nomp_finalize_logs();

#endif
