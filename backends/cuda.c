#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#define chk_cu(file, line, call)                                               \
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

#define BACKEND cuda
#define RUNTIME nvrtc
#define NOMP_BACKEND_FAILURE NOMP_CUDA_FAILURE

#define gpuDeviceProp_t cudaDeviceProp

#define gpuInit cuInit
#define gpuCtxCreate cuCtxCreate
#define gpuCtxDestroy cuCtxDestroy
#define gpuModuleLoadData cuModuleLoadData
#define gpuModuleGetFunction cuModuleGetFunction
#define gpuModuleLaunchKernel cuLaunchKernel
#define gpuModuleUnload cuModuleUnload

#define gpuModule CUmodule
#define gpuFunction CUfunction

#define GPURTC_SUCCESS NVRTC_SUCCESS
#define gpurtcGetCodeSize nvrtcGetPTXSize
#define gpurtcGetCode nvrtcGetPTX

#define check(call) chk_cu(__FILE__, __LINE__, call)

#include "unified-cuda-hip-impl.h"

#undef check

#undef gpurtcGetCode
#undef gpurtcGetCodeSize
#undef GPURTC_SUCCESS

#undef gpuFunction
#undef gpuModule

#undef gpuModuleUnload
#undef gpuModuleLaunchKernel
#undef gpuModuleGetFunction
#undef gpuModuleLoadData

#undef gpuDeviceProp

#undef NOMP_BACKEND_FAILURE
#undef RUNTIME
#undef BACKEND

#undef chk_cu
