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
  int errorno;
  nomp_log_type type;
};

static struct log *logs = NULL;
static unsigned logs_n = 0;
static unsigned logs_max = 0;
static const char *LOG_TYPE_STRING[] = {"Error", "Warning", "Info"};
static unsigned verbose = 0;

/**
 * @ingroup nomp_log_utils
 * @brief Set the verbose level for the log functions.
 *
 * @param[in] verbose_in Verbose level provided by the user.
 * @return int
 */
int nomp_log_set_verbose(const unsigned verbose_in) {
  verbose = verbose_in;
  return 0;
}

/**
 * @ingroup nomp_log_utils
 *
 * @brief Register a log with libnomp runtime.
 *
 * @details Register a log given a description of the log, error number and log
 * type. This function returns a unique id if the log type is an error and this
 * can be used to query the error log. If the log type is an information or a
 * warning, nomp_log() returns 0 and \p description is printed to stdout based
 * on the verbose level (which is set by either --nomp-verbose command line
 * argument or NOMP_VERBOSE environment variable) and not recorded by the
 * libnomp runtime. Also, the \p errorno is ignored if the log type is not an
 * error. On failure, nomp_log_() returns -1. Use nomp_log() macro without
 * calling this function directly.
 *
 * @param[in] description Detailed description of the log.
 * @param[in] errorno Log number which is defined in nomp.h
 * @param[in] type Type of the log (one of @ref nomp_log_type)
 * @return int
 */
int nomp_log_(const char *description, int errorno, nomp_log_type type, ...) {
  const char *type_str = LOG_TYPE_STRING[type - 1];
  size_t len = strlen(description) + strlen(type_str) + 10;
  char *desc = nomp_calloc(char, len);
  snprintf(desc, len, "[%s] %%s:%%u %s", type_str, description);

  char buf[BUFSIZ];
  va_list vargs;
  va_start(vargs, type);
  vsnprintf(buf, BUFSIZ, desc, vargs);
  va_end(vargs);
  nomp_free(&desc);

  // Print the logs based on the verbose level.
  if ((verbose >= NOMP_ERROR && type == NOMP_ERROR) ||
      (verbose >= NOMP_WARNING && type == NOMP_WARNING) ||
      (verbose >= NOMP_INFO && type == NOMP_INFO))
    printf("%s\n", buf);

  if (type == NOMP_ERROR) {
    if (logs_max <= logs_n) {
      logs_max += logs_max / 2 + 1;
      logs = nomp_realloc(logs, struct log, logs_max);
    }

    logs[logs_n].description = strndup(buf, BUFSIZ);
    logs[logs_n].errorno = errorno, logs[logs_n].type = type, logs_n++;
  }

  return type == NOMP_ERROR ? logs_n : 0;
}

/**
 * @ingroup nomp_user_api
 *
 * @brief Return the log given the log id.
 *
 * @details Returns the log of the given error id. Returns NULL if the
 * id is invalid.
 *
 * @param[in] id id of the error log returned by @ref nomp_log.
 * @return char*
 */
char *nomp_get_err_str(unsigned id) {
  if (id == 0 || id > logs_n)
    return NULL;

  return strndup(logs[id - 1].description, BUFSIZ);
}

/**
 * @ingroup nomp_user_api
 *
 * @brief Return log number given the log id.
 *
 * @details Returns the error number given the id. If id is invalid return
 * NOMP_USER_LOG_ID_IS_INVALID. Error number is one of @ref nomp_user_errors.
 *
 * @param[in] id id of the log returned by @ref nomp_log().
 * @return int
 */
int nomp_get_err_no(unsigned id) {
  if (id == 0 || id > logs_n)
    return NOMP_USER_LOG_ID_IS_INVALID;
  return logs[id - 1].errorno;
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
