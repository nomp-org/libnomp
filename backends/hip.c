#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>

const char *ERR_STR_BACKEND_FAILURE = "HIP %s failure: %s.";

#define BACKEND hip
#define RUNTIME hiprtc
#define NOMP_BACKEND_FAILURE NOMP_HIP_FAILURE

#define gpuDeviceProp_t hipDeviceProp_t

#define gpuInit hipInit
#define gpuCtxCreate hipCtxCreate
#define gpuCtxDestroy hipCtxDestroy
#define gpuModuleLoadData hipModuleLoadData
#define gpuModuleGetFunction hipModuleGetFunction
#define gpuModuleLaunchKernel hipModuleLaunchKernel
#define gpuModuleUnload hipModuleUnload

#define gpuModule hipModule_t
#define gpuFunction hipFunction_t

#define GPURTC_SUCCESS HIPRTC_SUCCESS
#define gpurtcGetCodeSize hiprtcGetCodeSize
#define gpurtcGetCode hiprtcGetCode

#define check(call) chk_gpu(call)

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
