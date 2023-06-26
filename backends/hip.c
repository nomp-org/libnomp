#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>

static const char *ERR_STR_BACKEND_FAILURE = "HIP %s failure: %s.";
#define NOMP_BACKEND_FAILURE NOMP_HIP_FAILURE

#define DRIVER hip
#define RUNTIME_COMPILATION hiprtc
#define RUNTIME hip

#define backendDeviceProp_t hipDeviceProp_t

#define backendModuleLaunchKernel hipModuleLaunchKernel

#define backendModule hipModule_t
#define backendFunction hipFunction_t

#define RTC_SUCCESS HIPRTC_SUCCESS
#define backendrtcGetCodeSize hiprtcGetCodeSize
#define backendrtcGetCode hiprtcGetCode

#define check_runtime(call) check_driver(call)

#include "unified-cuda-hip-impl.h"

#undef check_runtime

#undef backendrtcGetCode
#undef backendrtcGetCodeSize
#undef RTC_SUCCESS

#undef backendFunction
#undef backendModule

#undef backendModuleLaunchKernel

#undef backendDeviceProp

#undef RUNTIME
#undef RUNTIME_COMPILATION
#undef DRIVER

#undef NOMP_BACKEND_FAILURE
