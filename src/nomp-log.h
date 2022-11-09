#if !defined(_NOMP_LOG_H_)
#define _NOMP_LOG_H_
#include "nomp.h"
#include <stddef.h>

extern const char *ERR_STR_NOMP_IS_ALREADY_INITIALIZED;
extern const char *ERR_STR_FAILED_TO_INITIALIZE_NOMP;
extern const char *ERR_STR_FAILED_TO_FINALIZE_NOMP;
extern const char *ERR_STR_NOMP_INSTALL_DIR_NOT_SET;
extern const char *ERR_STR_NOMP_IS_NOT_INITIALIZED;

extern const char *ERR_STR_INVALID_MAP_OP;
extern const char *ERR_STR_INVALID_MAP_PTR;
extern const char *ERR_STR_PTR_IS_ALREADY_ALLOCATED;

extern const char *ERR_STR_KERNEL_RUN_FAILED;
extern const char *ERR_STR_INVALID_KERNEL;
extern const char *ERR_STR_NOMP_INVALID_CLAUSE;

extern const char *WARNING_STR_PYTHON_IS_ALREADY_INITIALIZED;
extern const char *ERR_STR_LOOPY_CONVERSION_ERROR;
extern const char *ERR_STR_FILE_NAME_NOT_PROVIDED;
extern const char *ERR_STR_USER_CALLBACK_NOT_PROVIDED;
extern const char *ERR_STR_USER_CALLBACK_NOT_FOUND;
extern const char *ERR_STR_USER_CALLBACK_FAILURE;
extern const char *ERR_STR_LOOPY_KNL_NAME_NOT_FOUND;
extern const char *ERR_STR_LOOPY_CODEGEN_FAILED;
extern const char *ERR_STR_LOOPY_GRIDSIZE_FAILED;
extern const char *ERR_STR_GRIDSIZE_CALCULATION_FAILED;
extern const char *ERR_STR_INVALID_KNL_ARG_TYPE;
extern const char *ERR_STR_INVALID_DEVICE;
extern const char *ERR_STR_KNL_ARG_SET_ERROR;
extern const char *ERR_STR_INVALID_PLATFORM;
extern const char *ERR_STR_MALLOC_ERROR;
extern const char *ERR_STR_KNL_BUILD_ERROR;
extern const char *ERR_STR_PY_INITIALIZE_ERROR;
extern const char *ERR_STR_INVALID_LOG_ID;
extern const char *ERR_STR_NOMP_UNKOWN_ERROR;
extern const char *ERR_STR_EXCEED_MAX_LEN_STR;
extern const char *ERR_STR_CUDA_FAILURE;
extern const char *ERR_STR_TCALLOC_FAILURE;
extern const char *ERR_STR_OPENCL_FAILURE;

int set_log_(const char *desc, int logno, nomp_log_type type, const char *fname,
             unsigned line_no, ...);
/**
 * @ingroup nomp_internal_api
 * @brief Sets a log.
 *
 * @details Sets a log given a description of the log and log type and returns a
 * unique log_id.
 * @param[in] logno unique id of the log kind.
 * @param[in] type type of the log either NOMP_ERROR, NOMP_WARNING or
 * NOMP_INFORMATION.
 * @param[in] desc detailed description of the log.
 * @return int
 */
#define set_log(logno, type, desc, ...)                                        \
  set_log_(desc, logno, type, __FILE__, __LINE__, ##__VA_ARGS__)

#endif
