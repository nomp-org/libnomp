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

#endif
