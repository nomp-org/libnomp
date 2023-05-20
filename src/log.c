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
static int verbose;

int nomp_log_init(const int verbose_in) {
  if (verbose_in < 0 || verbose_in > 3) {
    return nomp_set_log(NOMP_USER_INPUT_IS_INVALID, NOMP_ERROR,
                        "Invalid verbose level %u is provided. The value "
                        "should be within the range 0-3.",
                        verbose_in);
  }
  verbose = verbose_in;
  return 0;
}

int nomp_set_log_(const char *description, int logno, nomp_log_type type,
                  const char *fname, unsigned line, ...) {
  char buf[BUFSIZ];
  va_list vargs;
  va_start(vargs, line);
  vsnprintf(buf, BUFSIZ, description, vargs);
  va_end(vargs);

  char *file = strndup(fname, PATH_MAX);
  if (type < 0 || type > 2)
    return -1;
  const char *type_str = LOG_TYPE_STRING[type];

  // 10 for UINT_MAX, 5 for `[] : ` characters and 1 for `\0`.
  size_t len = strlen(buf) + strlen(file) + strlen(type_str) + 10 + 5 + 1;
  char *desc = nomp_calloc(char, len);
  snprintf(desc, len, "[%s] %s:%u %s", type_str, file, line, buf);

  // Print the logs based on the verbose level
  if ((verbose > 0 && type == NOMP_ERROR) ||
      (verbose > 1 && type == NOMP_WARNING) ||
      (verbose > 2 && type == NOMP_INFORMATION))
    printf("%s\n", desc);

  if (type == NOMP_ERROR) {
    if (logs_max <= logs_n) {
      logs_max += logs_max / 2 + 1;
      logs = nomp_realloc(logs, struct log, logs_max);
    }

    logs[logs_n].description = strndup(desc, len);
    logs[logs_n].logno = logno, logs[logs_n].type = type, logs_n++;
  }
  nomp_free(&desc), nomp_free(&file);

  return type == NOMP_ERROR ? logs_n : 0;
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

void nomp_logs_finalize() {
  for (unsigned i = 0; i < logs_n; i++)
    nomp_free(&logs[i].description);
  nomp_free(&logs), logs_n = logs_max = 0;
}

struct time_log {
  char *entry;
  int total_calls;
  long double total_time;
  long double last_call;
  clock_t last_tick;
};
static struct time_log *time_logs;
static unsigned time_logs_n = 0, time_logs_max = 0;

unsigned find_time_log(const char *entry) {
  for (int i = 0; i < time_logs_n; i++)
    if (strncmp(time_logs[i].entry, entry, NOMP_MAX_BUFSIZ) == 0)
      return i;
  return time_logs_n;
}

void nomp_profile(const char *name, const int toggle, const int profile_level,
                  const int sync) {
  if (profile_level == 0)
    return;

  if (toggle == 0 && sync == 1)
    nomp_sync();
  clock_t current_tick = clock();

  unsigned id = find_time_log(name);

  if (id == time_logs_n) { // Points to a new time log
    if (toggle == 1) {     // Starts the timer on a new time log
      // Dynamically increase the memory allocation
      if (time_logs_max <= time_logs_n) {
        time_logs_max += time_logs_max / 2 + 1;
        time_logs = nomp_realloc(time_logs, struct time_log, time_logs_max);
      }

      // Creates a new time_log
      time_logs[id].entry = strndup(name, NOMP_MAX_BUFSIZ);
      time_logs[id].total_calls = 0;
      time_logs[id].total_time = 0;
      time_logs[id].last_tick = current_tick;
      time_logs_n++;
    }
  } else if (toggle == 0) { // Turns off the timer
    // Ignore if the user toggles off the time by accident
    if (time_logs[id].last_tick == -1)
      return;

    // Captures the execution time.
    long double elapsed_time =
        (double)(current_tick - time_logs[id].last_tick) * 1000 /
        CLOCKS_PER_SEC;

    // Updates the current time log
    time_logs[id].last_call = elapsed_time;
    time_logs[id].total_calls++;
    time_logs[id].total_time += elapsed_time;
    time_logs[id].last_tick = -1;
  } else { // Starts the timer on an exiting time log
    time_logs[id].last_tick = current_tick;
  }
}

void nomp_profile_finalize() {
  for (int i = 0; i < time_logs_n; i++)
    nomp_free(&time_logs[i].entry);
  nomp_free(&time_logs), time_logs_n = time_logs_max = 0;
}

void nomp_profile_result() {
  printf("| %-24s | %12s | %18s | %18s | %18s |\n", "Entry", "Total Calls",
         "Total Time (ms)", "Last Call (ms)", "Average Time (ms)");
  printf("|--------------------------|--------------|--------------------|-----"
         "---------------|--------------------|\n");
  for (int i = 0; i < time_logs_n; i++) {
    long double avg_time = time_logs[i].total_time / time_logs[i].total_calls;
    printf("| %-24s | %12d | %18.4Lf | %18.4Lf | %18.4Lf |\n",
           time_logs[i].entry, time_logs[i].total_calls,
           time_logs[i].total_time, time_logs[i].last_call, avg_time);
  }
}
