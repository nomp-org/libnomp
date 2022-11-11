#if !defined(_NOMP_LOG_H_)
#define _NOMP_LOG_H_
#include "nomp.h"
#include <stddef.h>

extern const char *ERR_STR_NOMP_IS_ALREADY_INITIALIZED;
extern const char *ERR_STR_FAILED_TO_INITIALIZE_NOMP;
extern const char *ERR_STR_NOMP_IS_NOT_INITIALIZED;
extern const char *ERR_STR_FAILED_TO_FINALIZE_NOMP;

extern const char *ERR_STR_NOMP_INSTALL_DIR_NOT_SET;

extern const char *ERR_STR_MEM_ALLOC_FAILURE;

extern const char *ERR_STR_INVALID_DEVICE;
extern const char *ERR_STR_INVALID_PLATFORM;

extern const char *ERR_STR_INVALID_MAP_OP;
extern const char *ERR_STR_INVALID_MAP_PTR;
extern const char *ERR_STR_PTR_IS_ALREADY_ALLOCATED;

extern const char *ERR_STR_KERNEL_RUN_FAILED;
extern const char *ERR_STR_INVALID_KERNEL;
extern const char *ERR_STR_INVALID_KNL_ARG_TYPE;
extern const char *ERR_STR_KNL_ARG_SET_ERROR;
extern const char *ERR_STR_KNL_BUILD_ERROR;

extern const char *WARNING_STR_PYTHON_IS_ALREADY_INITIALIZED;
extern const char *ERR_STR_PY_INITIALIZE_ERROR;

extern const char *ERR_STR_LOOPY_CONVERSION_ERROR;
extern const char *ERR_STR_FILE_NAME_NOT_PROVIDED;
extern const char *ERR_STR_USER_CALLBACK_NOT_PROVIDED;
extern const char *ERR_STR_USER_CALLBACK_NOT_FOUND;
extern const char *ERR_STR_USER_CALLBACK_FAILURE;
extern const char *ERR_STR_NOMP_INVALID_CLAUSE;

extern const char *ERR_STR_LOOPY_KNL_NAME_NOT_FOUND;
extern const char *ERR_STR_LOOPY_CODEGEN_FAILED;
extern const char *ERR_STR_LOOPY_GRIDSIZE_FAILED;
extern const char *ERR_STR_GRIDSIZE_CALCULATION_FAILED;

extern const char *ERR_STR_INVALID_LOG_ID;
extern const char *ERR_STR_NOMP_UNKOWN_ERROR;

extern const char *ERR_STR_CUDA_FAILURE;
extern const char *ERR_STR_OPENCL_FAILURE;

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
