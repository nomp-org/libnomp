#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>

static const char *ERR_STR_BACKEND_FAILURE = "HIP %s failure: %s.";
static const char *FAST_MATH_FLAG          = "-ffast-math";

#define NOMP_BACKEND_FAILURE NOMP_HIP_FAILURE
#define NOMP_HIP             1

#define DRIVER              hip
#define RUNTIME             hip
#define RUNTIME_COMPILATION hiprtc
#define RTC_SUCCESS         HIPRTC_SUCCESS

#define backendDeviceProp_t       hipDeviceProp_t
#define backendModuleLaunchKernel hipModuleLaunchKernel
#define backendModule             hipModule_t
#define backendFunction           hipFunction_t

#define backendrtcGetCodeSize hiprtcGetCodeSize
#define backendrtcGetCode     hiprtcGetCode

#define check_runtime(call) check_driver(call)

#include "unified-cuda-hip-impl.h"

#undef check_runtime

#undef backendrtcGetCode
#undef backendrtcGetCodeSize

#undef backendFunction
#undef backendModule
#undef backendModuleLaunchKernel
#undef backendDeviceProp_t

#undef RTC_SUCCESS
#undef RUNTIME_COMPILATION
#undef RUNTIME
#undef DRIVER

#undef NOMP_HIP
#undef NOMP_BACKEND_FAILURE
