#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#define GPU cuda
#define RUNTIME nvrtc
#define GPU_CHECK chk_cu
#define NOMP_GPU_FAILURE NOMP_CUDA_FAILURE

#define GPURTC_SUCCESS NVRTC_SUCCESS
#define gpurtcGetCodeSize nvrtcGetPTXSize
#define gpurtcGetCode nvrtcGetPTX
#define gpuModuleLoadData cuModuleLoadData
#define gpuModuleGetFunction cuModuleGetFunction
#define gpuModuleLaunchKernel cuLaunchKernel
#define gpuModuleUnload cuModuleUnload

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

struct cuda_backend {
  int device_id;
  struct cudaDeviceProp prop;
};

struct cuda_prog {
  CUmodule module;
  CUfunction kernel;
};

#include "unified-cuda-hip-impl.h"
