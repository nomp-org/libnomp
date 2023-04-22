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

#define chk_cu(call)                                                           \
  chk_err_(__FILE__, __LINE__, call, CUresult, CUDA_SUCCESS,                   \
           cuGetErrorName(result, &msg), "operation");

static const char *ERR_STR_GPU_FAILURE = "Cuda %s failed: %s.";

#include "unified-cuda-hip-impl.h"

nvrtcResult cuda_compile(nvrtcProgram prog, struct cuda_backend *nbnd) {
  char arch[NOMP_MAX_BUFSIZ];
  snprintf(arch, NOMP_MAX_BUFSIZ, "-arch=compute_%d%d", nbnd->prop.major,
           nbnd->prop.minor);
  const char *opts[1] = {arch};
  return nvrtcCompileProgram(prog, 1, opts);
}

#undef GPU
#undef RUNTIME
#undef GPU_CHECK
#undef NOMP_GPU_FAILURE
#undef chk_cu

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
