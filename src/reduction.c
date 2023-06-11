#include "nomp-reduction.h"

#define NOMP_DO_SUM(a, b) (a) += (b)
#define NOMP_DO_PROD(a, b) (a) *= (b)

#define NOMP_INIT_SUM(a) (a) = 0
#define NOMP_INIT_PROD(a) (a) = 1

#define NOMP_REDUCTION(T, SUFFIX, OP)                                          \
  static void reduce##SUFFIX##_##OP(T *out, T *in, unsigned n) {               \
    NOMP_INIT_##OP(*out);                                                      \
    for (; n; n--)                                                             \
      NOMP_DO_##OP(*out, in[n - 1]);                                           \
  }

#define NOMP_FOR_EACH_DOMAIN(macro, OP)                                        \
  macro(int, _int, OP) macro(unsigned int, _uint, OP) macro(long, _long, OP)   \
      macro(unsigned long, _ulong, OP) macro(float, _float, OP)                \
          macro(double, _double, OP)

NOMP_FOR_EACH_DOMAIN(NOMP_REDUCTION, SUM)
NOMP_FOR_EACH_DOMAIN(NOMP_REDUCTION, PROD)

int nomp_host_side_reduction(struct nomp_backend *backend,
                             struct nomp_prog *prg, struct nomp_mem *m) {
  int dom = prg->reduction_type, op = prg->reduction_op;
  size_t size = prg->reduction_size;
  void *out = prg->reduction_ptr;

  nomp_check(backend->sync(backend));
  nomp_check(backend->update(backend, m, NOMP_FROM, 0, prg->global[0], size));

  // FIXME: Too much repetition.
  if (op == NOMP_SUM) {
    switch (dom) {
    case NOMP_INT:
      if (size == 4)
        reduce_int_SUM((int *)out, (int *)m->hptr, prg->global[0]);
      else
        reduce_long_SUM((long *)out, (long *)m->hptr, prg->global[0]);
      break;
    case NOMP_UINT:
      if (size == 4)
        reduce_uint_SUM((unsigned int *)out, (unsigned int *)m->hptr,
                        prg->global[0]);
      else
        reduce_ulong_SUM((unsigned long *)out, (unsigned long *)m->hptr,
                         prg->global[0]);
      break;
    case NOMP_FLOAT:
      if (size == 4)
        reduce_float_SUM((float *)out, (float *)m->hptr, prg->global[0]);
      else
        reduce_double_SUM((double *)out, (double *)m->hptr, prg->global[0]);
      break;
    }
  } else if (op == NOMP_PROD) {
    switch (dom) {
    case NOMP_INT:
      if (size == 4)
        reduce_int_PROD((int *)out, (int *)m->hptr, prg->global[0]);
      else
        reduce_long_PROD((long *)out, (long *)m->hptr, prg->global[0]);
      break;
    case NOMP_UINT:
      if (size == 4)
        reduce_uint_PROD((unsigned int *)out, (unsigned int *)m->hptr,
                         prg->global[0]);
      else
        reduce_ulong_PROD((unsigned long *)out, (unsigned long *)m->hptr,
                          prg->global[0]);
      break;
    case NOMP_FLOAT:
      if (size == 4)
        reduce_float_PROD((float *)out, (float *)m->hptr, prg->global[0]);
      else
        reduce_double_PROD((double *)out, (double *)m->hptr, prg->global[0]);
      break;
    }
  }

  return 0;
}
