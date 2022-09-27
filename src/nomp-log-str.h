#if !defined(_NOMP_LOG_STR_H_)
#define _NOMP_LOG_STR_H_

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

#endif
