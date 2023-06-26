#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#define check_cu(file, line, call)                                             \
  {                                                                            \
    CUresult result = (call);                                                  \
    if (result != CUDA_SUCCESS) {                                              \
      const char *msg;                                                         \
      cuGetErrorName(result, &msg);                                            \
      return nomp_log(NOMP_BACKEND_FAILURE, NOMP_ERROR,                        \
                      ERR_STR_BACKEND_FAILURE, "operation", msg);              \
    }                                                                          \
  }

static const char *ERR_STR_BACKEND_FAILURE = "CUDA %s failure: %s.";
#define NOMP_BACKEND_FAILURE NOMP_CUDA_FAILURE

#define DRIVER cuda
#define RUNTIME_COMPILATION nvrtc
#define RUNTIME cu

#define backendDeviceProp_t cudaDeviceProp

#define backendModuleLaunchKernel cuLaunchKernel

#define backendModule CUmodule
#define backendFunction CUfunction

#define RTC_SUCCESS NVRTC_SUCCESS
#define backendrtcGetCodeSize nvrtcGetPTXSize
#define backendrtcGetCode nvrtcGetPTX

#define check(call) check_cu(__FILE__, __LINE__, call)

#include "unified-cuda-hip-impl.h"

#undef check

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

#undef check_cu
