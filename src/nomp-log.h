#if !defined(_NOMP_LOG_H_)
#define _NOMP_LOG_H_
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

extern const char *WARNING_STR_PYTHON_IS_ALREADY_INITIALIZED;
extern const char *ERR_STR_LOOPY_CONVERSION_ERROR;
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

typedef enum {
  NOMP_ERROR = 0,
  NOMP_WARNING = 1,
  NOMP_INFORMATION = 2
} nomp_log_type;

struct log {
  char *description;
  int logno;
  nomp_log_type type;
};

static struct log *logs = NULL;
static unsigned logs_n = 0;
static unsigned logs_max = 0;

int nomp_set_log_(const char *desc, int logno, nomp_log_type type,
                  const char *fname, unsigned line_no, ...);
#define nomp_set_log(logno, type, desc, ...)                                   \
  nomp_set_log_(desc, logno, type, __FILE__, __LINE__, ##__VA_ARGS__);

/**
 * @ingroup nomp_user_api
 * @brief Return error description.
 *
 * @details Returns the error description given the error_id
 * @param[in] log variable to set the error description
 * @param[in] log_id id of the error
 * @param[in] type either NOMP_ERROR, NOMP_WARNING or NOMP_INFORMATION
 * @return int
 */
int nomp_get_log(char **log, int log_id, nomp_log_type type);

/**
 * @ingroup nomp_user_api
 * @brief Return error type.
 *
 * @details Returns the error_type given the error_id
 * @param[in] log_id id of the error
 * @return int
 */
int nomp_get_log_no(int log_id);
#endif
