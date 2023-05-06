#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>

#define GPU hip
#define RUNTIME hiprtc
#define GPU_CHECK chk_gpu
#define NOMP_GPU_FAILURE NOMP_HIP_FAILURE

#define GPURTC_SUCCESS HIPRTC_SUCCESS
#define gpurtcGetCodeSize hiprtcGetCodeSize
#define gpurtcGetCode hiprtcGetCode
#define gpuModuleLoadData hipModuleLoadData
#define gpuModuleGetFunction hipModuleGetFunction
#define gpuModuleLaunchKernel hipModuleLaunchKernel
#define gpuModuleUnload hipModuleUnload

#define GPU_COMPILE                                                            \
  const char *opts[1] = {NULL};                                                \
  hiprtcResult result = hiprtcCompileProgram(prog, 0, opts);

#define DESTROY_CTX                                                            \
  struct hip_backend *nbnd = (struct hip_backend *)bnd->bptr;                  \
  chk_gpu(hipCtxDestroy(nbnd->ctx));

#define CREATE_CTX                                                             \
  chk_gpu(hipInit(0));                                                         \
  chk_gpu(hipCtxCreate(&nbnd->ctx, 0, nbnd->device_id));

const char *ERR_STR_GPU_FAILURE = "HIP %s failed: %s.";

struct hip_backend {
  int device_id;
  struct hipDeviceProp_t prop;
  hipCtx_t ctx;
};

struct hip_prog {
  hipModule_t module;
  hipFunction_t kernel;
};

#include "unified-cuda-hip-impl.h"
