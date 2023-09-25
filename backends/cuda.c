#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

static const char *ERR_STR_BACKEND_FAILURE = "CUDA %s failure: %s.";
#define NOMP_BACKEND_FAILURE NOMP_CUDA_FAILURE
#define NOMP_CUDA 1

#define DRIVER cuda
#define RUNTIME_COMPILATION nvrtc
#define RUNTIME cu

#define backendDeviceProp_t struct cudaDeviceProp
#define backendModuleLaunchKernel cuLaunchKernel
#define backendModule CUmodule
#define backendFunction CUfunction

#define RTC_SUCCESS NVRTC_SUCCESS
#define backendrtcGetCodeSize nvrtcGetPTXSize
#define backendrtcGetCode nvrtcGetPTX

#define check_runtime(call)                                                    \
  {                                                                            \
    CUresult result = (call);                                                  \
    if (result != CUDA_SUCCESS) {                                              \
      const char *msg;                                                         \
      cuGetErrorName(result, &msg);                                            \
      return nomp_log(NOMP_BACKEND_FAILURE, NOMP_ERROR,                        \
                      ERR_STR_BACKEND_FAILURE, "operation", msg);              \
    }                                                                          \
  }

#include "unified-cuda-hip-impl.h"

#undef check_runtime

#undef backendrtcGetCode
#undef backendrtcGetCodeSize
#undef RTC_SUCCESS

#undef backendFunction
#undef backendModule
#undef backendModuleLaunchKernel
#undef backendDeviceProp_t

#undef RUNTIME
#undef RUNTIME_COMPILATION
#undef DRIVER

#undef NOMP_CUDA
#undef NOMP_BACKEND_FAILURE
