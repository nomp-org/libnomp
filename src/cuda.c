#include "nomp-impl.h"

static int cuda_map(struct backend *bnd, struct mem *m, const int op);
static void cuda_map_ptr(void **p, size_t *size, struct mem *m);
static int cuda_knl_build(struct backend *bnd, struct prog *prg,
                            const char *source, const char *name);
static int cuda_knl_set(struct prog *prg, const int index, const size_t size,
                          void *arg);
static int cuda_knl_run(struct backend *bnd, struct prog *prg, const int ndim,
                          const size_t *global, const size_t *local);
static int cuda_knl_free(struct prog *prg);
static int cuda_finalize(struct backend *bnd);
