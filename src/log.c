#include "nomp-impl.h"
#include <ctype.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const char *ERR_STR_RUNTIME_MEMORY_ALLOCATION_FAILURE =
    "libnomp host memory allocation failed.";

const char *ERR_STR_USER_MAP_PTR_IS_INVALID =
    "Map pointer %p was not found on device.";
const char *ERR_STR_USER_DEVICE_IS_INVALID =
    "Device id %d passed into libnomp is not valid.";

const char *ERR_STR_KNL_ARG_TYPE_IS_INVALID =
    "Invalid libnomp kernel argument type %d.";
const char *ERR_STR_KNL_ARG_SET_ERROR =
    "Setting libnomp kernel argument failed.";

const char *ERR_STR_USER_CALLBACK_NOT_PROVIDED =
    "User callback function is not provided.";
const char *ERR_STR_USER_CALLBACK_NOT_FOUND =
    "Specified user callback function not found in file %s.";
const char *ERR_STR_USER_CALLBACK_FAILURE = "User callback function %s failed.";

const char *ERR_STR_LOOPY_CONVERSION_ERROR = "C to Loopy conversion failed.";
const char *ERR_STR_LOOPY_KNL_NAME_NOT_FOUND =
    "Failed to find loopy kernel %s.";
const char *ERR_STR_LOOPY_CODEGEN_FAILED =
    "Code generation from loopy kernel %s failed.";
const char *ERR_STR_LOOPY_GRIDSIZE_FAILED = "Loopy grid size failure.";

const char *ERR_STR_CUDA_FAILURE = "Cuda %s failed: %s.";
const char *ERR_STR_OPENCL_FAILURE = "OpenCL %s failure with error code: %d.";

struct log {
  char *description;
  int logno;
  nomp_log_type type;
};

static struct log *logs = NULL;
static unsigned logs_n = 0, logs_max = 0;
static const char *LOG_TYPE_STRING[] = {"Error", "Warning", "Information"};

int set_log_(const char *description, int logno, nomp_log_type type,
             const char *fname, unsigned line, ...) {
  if (logs_max <= logs_n) {
    logs_max += logs_max / 2 + 1;
    logs = trealloc(logs, struct log, logs_max);
    if (logs == NULL)
      return NOMP_RUNTIME_MEMORY_ALLOCATION_FAILED;
  }

  va_list vargs;
  char buf[BUFSIZ];
  va_start(vargs, line);
  vsnprintf(buf, BUFSIZ, description, vargs);
  va_end(vargs);

  char *desc = strndup(buf, BUFSIZ), *file = strndup(fname, pathlen(fname));
  const char *type_str = LOG_TYPE_STRING[type];

  size_t len = strnlen(desc, BUFSIZ) + strnlen(file, BUFSIZ);
  len += strnlen(type_str, BUFSIZ) + 10 + 5 + 1; // 10 for UINT_MAX

  logs[logs_n].description = tcalloc(char, len);
  snprintf(logs[logs_n].description, len, "[%s] %s:%u %s", type_str, fname,
           line, desc);
  logs[logs_n].logno = logno, logs[logs_n].type = type, logs_n += 1;
  tfree(desc), tfree(file);

  return logs_n;
}

int nomp_get_log_str(char **log_str, int log_id) {
  if (log_id <= 0 || log_id > logs_n) {
    *log_str = NULL;
    return NOMP_USER_LOG_ID_IS_INVALID;
  }
  struct log lg = logs[log_id - 1];
  size_t n_desc = strnlen(lg.description, BUFSIZ) + 1;
  *log_str = tcalloc(char, n_desc);
  strncpy(*log_str, lg.description, n_desc);
  return 0;
}

int nomp_get_log_no(int log_id) {
  if (log_id <= 0 || log_id > logs_n)
    return NOMP_USER_LOG_ID_IS_INVALID;
  return logs[log_id - 1].logno;
}

nomp_log_type nomp_get_log_type(int log_id) {
  if (log_id <= 0 || log_id > logs_n)
    return NOMP_INVALID;
  return logs[log_id - 1].type;
}

void nomp_finalize_logs() {
  for (unsigned i = 0; i < logs_n; i++)
    tfree(logs[i].description);
  tfree(logs);
  logs = NULL, logs_n = logs_max = 0;
}
