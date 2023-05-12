#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>

#define GPU hip
#define RUNTIME hiprtc
#define GPU_CHECK chk_gpu
#define NOMP_GPU_FAILURE NOMP_HIP_FAILURE

#define gpuDeviceProp hipDeviceProp_t
#define gpuCtx hipCtx_t
#define gpuModule hipModule_t
#define gpuFunction hipFunction_t
#define GPURTC_SUCCESS HIPRTC_SUCCESS
#define gpurtcGetCodeSize hiprtcGetCodeSize
#define gpurtcGetCode hiprtcGetCode
#define gpuModuleLoadData hipModuleLoadData
#define gpuModuleGetFunction hipModuleGetFunction
#define gpuModuleLaunchKernel hipModuleLaunchKernel
#define gpuModuleUnload hipModuleUnload
#define gpuInit hipInit
#define gpuCtxCreate hipCtxCreate
#define gpuCtxDestroy hipCtxDestroy

const char *ERR_STR_GPU_FAILURE = "HIP %s failed: %s.";

#include "unified-cuda-hip-impl.h"

hiprtcResult hip_compile(hiprtcProgram prog, struct hip_backend *nbnd) {
  const char *opts[1] = {NULL};
  return hiprtcCompileProgram(prog, 0, opts);
}

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
