#include "nomp-impl.h"
#include <ctype.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const char *ERR_STR_USER_MAP_PTR_IS_INVALID =
    "Map pointer %p was not found on device.";
const char *ERR_STR_USER_DEVICE_IS_INVALID =
    "Device id %d passed into libnomp is not valid.";

struct log {
  char *description;
  int logno;
  nomp_log_type type;
};

static struct log *logs = NULL;
static unsigned logs_n = 0, logs_max = 0;
static const char *LOG_TYPE_STRING[] = {"Error", "Warning", "Info"};
static int nomp_verbose_level;

int nomp_log_init(const int user_verbose) {
  if (user_verbose < 0 || user_verbose > 3)
    return nomp_set_log(NOMP_USER_INPUT_IS_INVALID, NOMP_ERROR,
                        "Invalid verbose level %u is provided. The value "
                        "should be within the range 0-3.",
                        user_verbose);
  nomp_verbose_level = user_verbose;
  return 0;
}

int nomp_set_log_(const char *description, int logno, nomp_log_type type,
                  const char *fname, unsigned line, ...) {
  if (logs_max <= logs_n) {
    logs_max += logs_max / 2 + 1;
    logs = nomp_realloc(logs, struct log, logs_max);
  }

  char buf[BUFSIZ];

  va_list vargs;
  va_start(vargs, line);
  vsnprintf(buf, BUFSIZ, description, vargs);
  va_end(vargs);

  char *desc = strndup(buf, BUFSIZ), *file = strndup(fname, PATH_MAX);
  const char *type_str = LOG_TYPE_STRING[type];

  // 10 for UINT_MAX, 5 for `[] : ` characters and 1 for `\0`.
  size_t len = strlen(desc) + strlen(file) + strlen(type_str) + 10 + 5 + 1;
  logs[logs_n].description = nomp_calloc(char, len);
  snprintf(logs[logs_n].description, len, "[%s] %s:%u %s", type_str, fname,
           line, desc);

  // Print the logs based on the verbose level
  int print_log = (nomp_verbose_level > 0 && type == NOMP_ERROR) ||
                  (nomp_verbose_level > 1 && type == NOMP_WARNING) ||
                  (nomp_verbose_level > 2 && type == NOMP_INFORMATION);
  if (print_log)
    printf("%s\n", logs[logs_n].description);

  nomp_free(desc), nomp_free(file);
  logs[logs_n].logno = logno, logs[logs_n].type = type, logs_n++;

  return logs_n;
}

char *nomp_get_log_str(int id) {
  if (id <= 0 || id > logs_n)
    return NULL;

  return strndup(logs[id - 1].description, BUFSIZ);
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
    nomp_free(logs[i].description);
  nomp_free(logs);
  logs = NULL, logs_n = logs_max = 0;
}
