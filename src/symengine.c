#include "nomp-impl.h"
#include <symengine/cwrapper.h>

int nomp_symengine_update(CMapBasicBasic *map, const char *key,
                          const long val) {
  basic a, b, c;
  basic_new_stack(a), basic_new_stack(b), basic_new_stack(c);
  symbol_set(a, key), integer_set_si(b, val);
  int eval_grid = 0;
  if (!mapbasicbasic_get(map, a, c) || !basic_eq(c, b))
    mapbasicbasic_insert(map, a, b), eval_grid = 1;
  basic_free_stack(a), basic_free_stack(b), basic_free_stack(c);

  return eval_grid;
}

static int symengine_evaluate(size_t *out, unsigned i, CVecBasic *vec,
                              CMapBasicBasic *map) {
  basic a;
  basic_new_stack(a);
  vecbasic_get(vec, i, a);
  CWRAPPER_OUTPUT_TYPE err = basic_subs(a, a, map);
  if (err) {
    return nomp_log(
        NOMP_LOOPY_GRIDSIZE_FAILURE, NOMP_ERROR,
        "Expression substitute with SymEngine failed with error %d.", err);
  }
  out[i] = integer_get_ui(a);
  basic_free_stack(a);

  return 0;
}

int nomp_symengine_eval_grid_size(struct nomp_prog_t *prg) {
  // If the expressions are not NULL, iterate through them and evaluate with
  // pymbolic. Also, we should calculate and store a hash of the dict that
  // is passed. If the hash is the same, no need of re-evaluating the grid
  // size.
  for (unsigned i = 0; i < 3; i++)
    prg->global[i] = prg->local[i] = 1;

  for (unsigned j = 0; j < vecbasic_size(prg->sym_global); j++)
    nomp_check(symengine_evaluate(prg->global, j, prg->sym_global, prg->map));

  for (unsigned j = 0; j < vecbasic_size(prg->sym_local); j++)
    nomp_check(symengine_evaluate(prg->local, j, prg->sym_local, prg->map));

  for (unsigned i = 0; i < 3; i++)
    prg->gws[i] = prg->global[i] * prg->local[i];

  return 0;
}
