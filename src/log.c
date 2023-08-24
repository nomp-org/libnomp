#include "nomp-impl.h"
#include <stdarg.h>
#include <stdio.h>
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
static unsigned logs_n = 0;
static unsigned logs_max = 0;
static const char *LOG_TYPE_STRING[] = {"Error", "Warning", "Info"};
static int verbose = 0;

/**
 * @ingroup nomp_log_utils
 * @brief Set the verbose level for the log functions.
 *
 * @param[in] verbose_in Verbose level provided by the user.
 * @return int
 */
int nomp_log_set_verbose(const int verbose_in) {
  if (verbose_in < 0 || verbose_in > 3) {
    return nomp_log(NOMP_USER_INPUT_IS_INVALID, NOMP_ERROR,
                    "Invalid verbose level %u is provided. The value "
                    "should be within the range 0-3.",
                    verbose_in);
  }
  verbose = verbose_in;
  return 0;
}

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
 * @param[in] description Detailed description of the log.
 * @param[in] logno Log number which is defined in nomp.h
 * @param[in] type Type of the log (one of @ref nomp_log_type)
 * @param[in] fname File name in which the nomp_log_() is called.
 * @param[in] line Line number where the nomp_log_() is called.
 * @return int
 */
int nomp_log_(const char *description, int logno, nomp_log_type type,
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

  // Print the logs based on the verbose level.
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

/**
 * @ingroup nomp_user_api
 * @brief Return the log description given the log id.
 *
 * @details Returns the log description given the log id. Returns NULL if the
 * id is invalid.
 * @param[in] id id of the error.
 * @return char*
 */
char *nomp_get_log_str(int id) {
  if (id <= 0 || id > (int)logs_n)
    return NULL;

  return strndup(logs[id - 1].description, BUFSIZ);
}

/**
 * @ingroup nomp_user_api
 * @brief Return log number.
 *
 * @details Returns the log number given the log_id. If log_id
 * is invalid return NOMP_USER_LOG_ID_IS_INVALID.
 * @param[in] log_id id of the log.
 * @return int
 */
int nomp_get_log_no(int log_id) {
  if (log_id <= 0 || log_id > (int)logs_n)
    return NOMP_USER_LOG_ID_IS_INVALID;
  return logs[log_id - 1].logno;
}

/**
 * @ingroup nomp_user_api
 * @brief Return log type.
 *
 * @details Returns the log type given the log_id. Log type is either
 * NOMP_ERROR, NOMP_INFORMATION or NOMP_WARNING. If log_id is invalid return
 * NOMP_INVALID.
 * @param[in] log_id id of the log.
 * @return int
 */
nomp_log_type nomp_get_log_type(int log_id) {
  if (log_id <= 0 || log_id > (int)logs_n)
    return NOMP_INVALID;
  return logs[log_id - 1].type;
}

/**
 * @ingroup nomp_log_utils
 * @brief Free variables used to keep track of logs.
 *
 * @return void
 */
void nomp_log_finalize(void) {
  for (unsigned i = 0; i < logs_n; i++)
    nomp_free(&logs[i].description);
  nomp_free(&logs), logs_n = logs_max = 0;
}

struct time_log {
  char *entry;
  unsigned total_calls;
  double total_time;
  double last_call;
  clock_t last_tick;
};

static struct time_log *time_logs = NULL;
static unsigned time_logs_n = 0;
static unsigned time_logs_max = 0;
static int profile_level = 0;

/**
 * @ingroup nomp_profiler_utils
 * @brief Set the profile level for the nomp profiler.
 *
 * @param[in] profile_level_in Profile level provided by the user.
 * @return int
 */
int nomp_profile_set_level(const int profile_level_in) {
  profile_level = profile_level_in;
  return 0;
}

static unsigned find_time_log(const char *entry) {
  for (unsigned i = 0; i < time_logs_n; i++) {
    if (strncmp(time_logs[i].entry, entry, NOMP_MAX_BUFSIZ) == 0)
      return i;
  }
  return time_logs_n;
}

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
void nomp_profile(const char *name, const int toggle, const int sync) {
  if (profile_level == 0)
    return;

  if (toggle == 0 && sync == 1)
    nomp_sync();
  clock_t current_tick = clock();

  unsigned id = find_time_log(name);
  // If the timer is not found, create a new one.
  if (id == time_logs_n) {
    // Starts the timer if toggle is on.
    if (toggle == 1) {
      // Dynamically increase the memory if needed.
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
  } else if (toggle == 0) {
    // Ignore if the user toggles off the timer by accident.
    if (time_logs[id].last_tick == 0)
      return;

    // Captures the execution time.
    double elapsed =
        (double)(current_tick - time_logs[id].last_tick) / CLOCKS_PER_SEC;
    elapsed *= 1000.0;

    // Updates the current timing information.
    time_logs[id].last_call = elapsed;
    time_logs[id].total_time += elapsed;
    time_logs[id].last_tick = 0;
    time_logs[id].total_calls++;
  } else {
    // Toggle is on. So, we start the timer on an exiting time log.
    time_logs[id].last_tick = current_tick;
  }
}

/**
 * @ingroup nomp_profiler_utils
 * @brief Prints all the execution times recorded by the program.
 * This function is executed only when the `--nomp-profile` is provided.
 *
 * @return int
 */
void nomp_profile_result(void) {
  if (profile_level == 0)
    return;

  printf("| %-24s | %12s | %18s | %18s | %18s |\n", "Entry", "Total Calls",
         "Total Time (ms)", "Last Call (ms)", "Average Time (ms)");
  printf("|--------------------------|--------------|--------------------|-----"
         "---------------|--------------------|\n");
  for (unsigned i = 0; i < time_logs_n; i++) {
    double avg_time = time_logs[i].total_time / time_logs[i].total_calls;
    printf("| %-24s | %12d | %18.4lf | %18.4lf | %18.4lf |\n",
           time_logs[i].entry, time_logs[i].total_calls,
           time_logs[i].total_time, time_logs[i].last_call, avg_time);
  }
}

/**
 * @ingroup nomp_profiler_utils
 * @brief Free variables used to keep track of time logs.
 *
 * @return void
 */
void nomp_profile_finalize(void) {
  for (unsigned i = 0; i < time_logs_n; i++)
    nomp_free(&time_logs[i].entry);
  nomp_free(&time_logs), time_logs_n = time_logs_max = 0;
}
