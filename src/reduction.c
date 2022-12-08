#include "nomp-impl.h"

#define NOMP_DO_SUM(a, b) (a) += (b)
#define NOMP_DO_PROD(a, b) (a) *= (b)

#define NOMP_INIT_SUM(a) (a) = 0
#define NOMP_INIT_PROD(a) (a) = 1

#define NOMP_FOR_EACH_OP(T, macro) macro(T, SUM) macro(T, PROD)

#define NOMP_FOR_EACH_DOMAIN(macro) macro(int) macro(float)

#define NOMP_REDUCTION(T, OP)                                                  \
  static void reduce_##T##_##OP(T *out, T *in, unsigned n) {                   \
    NOMP_INIT_##OP(*out);                                                      \
    for (; n; n--)                                                             \
      NOMP_DO_##OP(*out, in[n - 1]);                                           \
  }

#define DEFINE1(T) NOMP_FOR_EACH_OP(T, NOMP_REDUCTION)
#define DEFINE_REDUCTION NOMP_FOR_EACH_DOMAIN(DEFINE1)
DEFINE_REDUCTION
#undef DEFINE_REDUCTION
#undef DEFINE1

// FIXME: Temporary hack -- should be fixed
#define NOMP_int NOMP_INT
#define NOMP_float NOMP_FLOAT

#define NOMP_SWITCH_OP_CASE(T, OP)                                             \
  case NOMP_##OP:                                                              \
    WITH_OP(T, OP);                                                            \
    break;
#define NOMP_SWITCH_OP(T, OP)                                                  \
  switch (op) {                                                                \
    NOMP_FOR_EACH_OP(T, NOMP_SWITCH_OP_CASE)                                   \
  default:                                                                     \
    break;                                                                     \
  }

#define NOMP_SWITCH_DOMAIN_CASE(T)                                             \
  case NOMP_##T:                                                               \
    WITH_DOMAIN(T);                                                            \
    break;

#define NOMP_SWITCH_DOMAIN(dom)                                                \
  do {                                                                         \
    switch (dom) {                                                             \
      NOMP_FOR_EACH_DOMAIN(NOMP_SWITCH_DOMAIN_CASE)                            \
    default:                                                                   \
      break;                                                                   \
    }                                                                          \
  } while (0)

int host_side_reduction(struct backend *nomp, struct prog *prg, struct mem *m) {
  int err = nomp->update(nomp, m, NOMP_FROM);
  return_on_err(err);

  struct arg *args = prg->args;
  int i = prg->reduction_indx, op = prg->reduction_op;

#define WITH_OP(T, OP) reduce_##T##_##OP(args[i].hptr, m->hptr, prg->global[0])
#define WITH_DOMAIN(T) NOMP_SWITCH_OP(T, OP)
  NOMP_SWITCH_DOMAIN(args[i].type);
#undef WITH_DOMAIN
#undef WITH_OP

  return 0;
}
