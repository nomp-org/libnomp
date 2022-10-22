#include "nomp-impl.h"
#include <ctype.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const char *ERR_STR_NOMP_IS_ALREADY_INITIALIZED =
    "libnomp is already initialized to use %s. Call nomp_finalize() before "
    "calling nomp_init() again.";
const char *ERR_STR_FAILED_TO_INITIALIZE_NOMP =
    "Failed to initialize libnomp. Invalid backend: %s";
const char *ERR_STR_FAILED_TO_FINALIZE_NOMP = "Failed to finalize libnomp.";
const char *ERR_STR_NOMP_INSTALL_DIR_NOT_SET =
    "Environment variable NOMP_INSTALL_DIR, which is required by libnomp is "
    "not set.";
const char *ERR_STR_NOMP_IS_NOT_INITIALIZED = "Nomp is not initialized.";

const char *ERR_STR_INVALID_MAP_OP = "Invalid map pointer operation %d.";
const char *ERR_STR_INVALID_MAP_PTR = "Invalid map pointer %p.";
const char *ERR_STR_PTR_IS_ALREADY_ALLOCATED =
    "Pointer %p is already allocated on device.";

const char *ERR_STR_KERNEL_RUN_FAILED = "Kernel %d run failed";
const char *ERR_STR_INVALID_KERNEL = "Invalid kernel %d.";

const char *WARNING_STR_PYTHON_IS_ALREADY_INITIALIZED =
    "Python is already initialized. Using already initialized python version.";

const char *ERR_STR_LOOPY_CONVERSION_ERROR = "C to Loopy conversion failed.";
const char *ERR_STR_USER_CALLBACK_NOT_FOUND =
    "Specified user callback function not found in file %s.";
const char *ERR_STR_USER_CALLBACK_FAILURE = "User callback function %s failed.";
const char *ERR_STR_LOOPY_KNL_NAME_NOT_FOUND =
    "Failed to find loopy kernel %s.";
const char *ERR_STR_LOOPY_CODEGEN_FAILED =
    "Code generation from loopy kernel %s failed.";
const char *ERR_STR_LOOPY_GRIDSIZE_FAILED = "Loopy grid size failure.";
const char *ERR_STR_GRIDSIZE_CALCULATION_FAILED =
    "Loopy grid size calculation failure.";

const char *ERR_STR_INVALID_KNL_ARG_TYPE =
    "Invalid NOMP kernel argument type %d.";
const char *ERR_STR_INVALID_DEVICE = "Invalid NOMP device id %d.";
const char *ERR_STR_KNL_ARG_SET_ERROR = "Setting NOMP kernel argument failed.";
const char *ERR_STR_INVALID_PLATFORM = "Invalid NOMP platform id %d.";
const char *ERR_STR_MALLOC_ERROR = "NOMP malloc error.";
const char *ERR_STR_KNL_BUILD_ERROR = "NOMP kernel build error.";
const char *ERR_STR_PY_INITIALIZE_ERROR = "NOMP python initialize error.";
const char *ERR_STR_INVALID_LOG_ID = "Invalid log id %d.";
const char *ERR_STR_NOMP_UNKOWN_ERROR = "Unkown error id %d";

struct log {
  char *description;
  int logno;
  nomp_log_type type;
};

static struct log *logs = NULL;
static unsigned logs_n = 0, logs_max = 0;
static const char *LOG_TYPE_STRING[] = {"Error", "Warning", "Information"};

int nomp_set_log_(const char *description, int logno, nomp_log_type type,
                  const char *fname, unsigned line_no, ...) {
  if (logs_max <= logs_n) {
    logs_max += logs_max / 2 + 1;
    logs = (struct log *)realloc(logs, sizeof(struct log) * logs_max);
    if (logs == NULL)
      return NOMP_OUT_OF_MEMORY;
  }

  va_list vargs;
  char buf[BUFSIZ];
  va_start(vargs, line_no);
  vsnprintf(buf, BUFSIZ, description, vargs);
  va_end(vargs);

  const char *log_type_string = LOG_TYPE_STRING[type];
  size_t n_desc = strnlen(buf, BUFSIZ), n_file = strnlen(fname, BUFSIZ),
         n_log_type = strnlen(log_type_string, BUFSIZ);
  logs[logs_n].description =
      (char *)calloc(n_desc + n_file + n_log_type + 6 + 3, sizeof(char));
  snprintf(logs[logs_n].description, BUFSIZ + n_file + n_log_type + 6 + 3 + 1,
           "[%s] %s:%u %s", log_type_string, fname, line_no, buf);
  logs[logs_n].logno = logno;
  logs[logs_n].type = type;
  logs_n += 1;
  return logs_n;
}

int nomp_get_log(char **log_str, int log_id) {
  if (log_id <= 0 && log_id > logs_n) {
    *log_str = NULL;
    return nomp_set_log(NOMP_INVALID_LOG_ID, NOMP_ERROR, ERR_STR_INVALID_LOG_ID,
                        log_id);
  }
  struct log lg = logs[log_id - 1];
  size_t n_desc = strnlen(lg.description, BUFSIZ) + 1;
  *log_str = (char *)calloc(n_desc, sizeof(char));
  strncpy(*log_str, lg.description, n_desc);
  return 0;
}

int nomp_get_log_no(int log_id) {
  if (log_id <= 0 && log_id > logs_n)
    return nomp_set_log(NOMP_INVALID_LOG_ID, NOMP_ERROR, ERR_STR_INVALID_LOG_ID,
                        log_id);
  return logs[log_id - 1].logno;
}

void nomp_finalize_logs() {
  for (unsigned i = 0; i < logs_n; i++)
    FREE(logs[i].description);
  FREE(logs);
  logs = NULL, logs_n = logs_max = 0;
}
