#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#define GPU cuda
#define RUNTIME nvrtc
#define GPU_CHECK chk_cu
#define NOMP_GPU_FAILURE NOMP_CUDA_FAILURE

#define gpuDeviceProp cudaDeviceProp
#define gpuCtx CUcontext
#define gpuModule CUmodule
#define gpuFunction CUfunction
#define GPURTC_SUCCESS NVRTC_SUCCESS
#define gpurtcGetCodeSize nvrtcGetPTXSize
#define gpurtcGetCode nvrtcGetPTX
#define gpuModuleLoadData cuModuleLoadData
#define gpuModuleGetFunction cuModuleGetFunction
#define gpuModuleLaunchKernel cuLaunchKernel
#define gpuModuleUnload cuModuleUnload
#define gpuInit cuInit
#define gpuCtxCreate cuCtxCreate
#define gpuCtxDestroy cuCtxDestroy

#define GPU_COMPILE                                                            \
  char arch[MAX_BUFSIZ];                                                       \
  snprintf(arch, MAX_BUFSIZ, "-arch=compute_%d%d", nbnd->prop.major,           \
           nbnd->prop.minor);                                                  \
  const char *opts[1] = {arch};                                                \
  nvrtcResult result = nvrtcCompileProgram(prog, 1, opts);

#define chk_cu(call)                                                           \
  {                                                                            \
    CUresult x = (call);                                                       \
    if (x != CUDA_SUCCESS) {                                                   \
      const char *msg;                                                         \
      cuGetErrorName(x, &msg);                                                 \
      return nomp_set_log(NOMP_CUDA_FAILURE, NOMP_ERROR, ERR_STR_GPU_FAILURE,  \
                          "CU operation", msg);                                \
    }                                                                          \
  }

static const char *ERR_STR_GPU_FAILURE = "Cuda %s failed: %s.";

#include "unified-cuda-hip-impl.h"

#undef GPU
#undef RUNTIME
#undef GPU_CHECK
#undef NOMP_GPU_FAILURE

#undef gpuDeviceProp
#undef gpuCtx
#undef gpuModule
#undef gpuFunction
#undef GPURTC_SUCCESS
#undef gpurtcGetCodeSize
#undef gpurtcGetCode
#undef gpuModuleLoadData
#undef gpuModuleGetFunction
#undef gpuModuleLaunchKernel
#undef gpuModuleUnload
#undef gpuInit
#undef gpuCtxCreate
#undef gpuCtxDestroy
