#if !defined(_NOMP_LOG_H_)
#define _NOMP_LOG_H_

#include <stddef.h>

#include "nomp.h"

extern const char *ERR_STR_USER_MAP_PTR_IS_INVALID;
extern const char *ERR_STR_USER_DEVICE_IS_INVALID;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup nomp_log_utils Internal functions used by logging module.
 */

/**
 * @ingroup nomp_log_utils
 * @brief Set the verbose level for the log functions.
 *
 * @param[in] verbose Verbose level provided by the user.
 * @return int
 */
int nomp_log_set_verbose(const int verbose);

/**
 * @ingroup nomp_log_utils
 * @brief Register a log with libnomp runtime.
 *
 * @details Register a log given a description of the log, log number and log
 * type. Returns a unique log id in case of errors which can be used to query
 * log later on success. In case of information or warnings, nomp_log()
 * returns 0 and details are printed to stdout based on the verbose level which
 * is either set by --nomp-verbose command line argument or NOMP_VERBOSE
 * environment variable. On failure, nomp_log_() returns -1. Use
 * nomp_log() macro to by pass the arguments \p fname and \p line_no.
 *
 * @param[in] logno Log number which is defined in nomp.h
 * @param[in] type Type of the log (one of @ref nomp_log_type)
 * @param[in] desc Detailed description of the log.
 * @param[in] fname File name in which the nomp_log_() is called.
 * @param[in] line_no Line number where the nomp_log_() is called.
 * @return int
 */
int nomp_log_(const char *desc, int logno, nomp_log_type type,
              const char *fname, unsigned line_no, ...);

#define NOMP_CASE_IMPL(_1, _2, _3, _4, _5, _6, _7, _8, N, ...) N
#define NOMP_CASE(...) NOMP_CASE_IMPL(__VA_ARGS__, 2, 2, 2, 2, 2, 2, 2, 1, 0)

#define NOMP_FIRST_IMPL(first, ...) first
#define NOMP_FIRST(...) NOMP_FIRST_IMPL(__VA_ARGS__, throwaway)

#define NOMP_REST_IMPL_WITH_2(first, ...) , __VA_ARGS__
#define NOMP_REST_IMPL_WITH_1(first)
#define NOMP_REST_IMPL_(num, ...) NOMP_REST_IMPL_WITH_##num(__VA_ARGS__)
#define NOMP_REST_IMPL(num, ...) NOMP_REST_IMPL_(num, __VA_ARGS__)
#define NOMP_REST(...) NOMP_REST_IMPL(NOMP_CASE(__VA_ARGS__), __VA_ARGS__)

#define nomp_log(logno, type, ...)                                             \
  nomp_log_(NOMP_FIRST(__VA_ARGS__), logno, type, __FILE__,                    \
            __LINE__ NOMP_REST(__VA_ARGS__))

/**
 * @ingroup nomp_log_utils
 * @brief Free variables used to keep track of logs.
 *
 * @return void
 */
void nomp_log_finalize(void);

/**
 * @defgroup nomp_profiler_utils Internal functions for profiling a
 * Nomp program.
 */

/**
 * @ingroup nomp_profiler_utils
 * @brief Set the profile level for the nomp profiler.
 *
 * @param[in] profile_level Profile level provided by the user.
 * @return int
 */
int nomp_profile_set_level(const int profile_level);

/**
 * @ingroup nomp_profiler_utils
 * @brief Toggles the timer and records the execution time between the two
 * consecutive uses of the function.
 *
 * @details The function either starts or ends the timer by considering the
 * toggle value. The function will start the timer if the toggle is 1. Else,
 * it will capture the execution time and records in a log.
 * @code{.c}
 * nomp_profile("Entry Name", 1, nomp.profile, 1);
 * // Code to be measured
 * nomp_profile("Entry Name", 0, nomp.profile, 1);
 * @endcode
 *
 * @param[in] name Name of the execution time that is being profiled.
 * @param[in] toggle Toggles the timer between tick (start of timing) and a tock
 * (end of timing).
 * @param[in] sync Execute nomp_sync when toggling off the timer.
 * @return void
 */
void nomp_profile(const char *name, int toggle, int sync);

/**
 * @ingroup nomp_profiler_utils
 * @brief Prints all the execution times recorded by the program.
 * This function is executed only when the `--nomp-profile` is provided.
 *
 * @return int
 */
void nomp_profile_result(void);

/**
 * @ingroup nomp_profiler_utils
 * @brief Free variables used to keep track of time logs.
 *
 * @return void
 */
void nomp_profile_finalize(void);

#ifdef __cplusplus
}
#endif

#endif // _NOMP_LOG_H_
