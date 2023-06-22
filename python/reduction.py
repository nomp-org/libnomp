"""Module to do reductions with using Loopy."""

from typing import Dict, List

import loopy as lp
import pymbolic.mapper
import pymbolic.primitives as prim
from loopy.symbolic import Reduction
from loopy.transform.data import reduction_arg_to_subst_rule
from loopy_api import LOOPY_INSN_PREFIX, LOOPY_LANG_VERSION

_TARGET_BLOCK_SIZE = {"cuda": 32, "opencl": 32, "sycl": 32, "hip": 32}


class InameCollector(pymbolic.mapper.WalkMapper):
    """Get all the inames in a pymbolic expression."""

    def __init__(self, expr):
        self.inames = []
        self.rec(expr)

    def map_variable(self, expr, *args, **kwargs):
        self.inames.append(expr.name)
        super().map_variable(expr, args, kwargs)

    def get_inames(self) -> List[str]:
        """Returns the inames which were found."""
        return self.inames

    def map_algebraic_leaf(self, expr, *args, **kwargs):
        raise NotImplementedError


def realize_reduction(
    tunit: lp.translation_unit.TranslationUnit,
    var: str,
    context: Dict[str, str],
) -> lp.translation_unit.TranslationUnit:
    """Perform transformations to realize a reduction."""

    if len(tunit.callables_table.keys()) > 1:
        raise NotImplementedError(
            "Don't know how to handle more than 1 callable in translation unit!"
        )

    knl = tunit.default_entrypoint
    if len(knl.inames.keys()) > 1:
        raise NotImplementedError(
            "Don't know how to handle more than 1 iname in redcution !"
        )

    (iname,) = knl.inames
    i_inner, i_outer = f"{iname}_inner", f"{iname}_outer"
    tunit = lp.split_iname(
        tunit,
        iname,
        _TARGET_BLOCK_SIZE[context["backend::name"]],
        inner_iname=i_inner,
        outer_iname=i_outer,
    )

    (knl_name,) = tunit.callables_table.keys()

    knl, insns, precompute = tunit[knl_name], [], 0
    for insn in knl.instructions:
        if (
            isinstance(insn, lp.Assignment)
            and isinstance(insn.assignee, prim.Subscript)
            and isinstance(insn.assignee.aggregate, prim.Variable)
            and insn.assignee.aggregate.name == var
        ):
            (lhs, *rhs) = insn.expression.children
            if i_inner in InameCollector(*rhs).get_inames():
                precompute = 1
            if isinstance(insn.expression, prim.Sum):
                rhs = Reduction(
                    lp.library.reduction.SumReductionOperation(),
                    i_inner,
                    prim.Sum(tuple(rhs)),
                )
            elif isinstance(insn.expression, prim.Product):
                rhs = Reduction(
                    lp.library.reduction.ProductReductionOperation(),
                    i_inner,
                    prim.Product(tuple(rhs)),
                )

            if isinstance(lhs, prim.Subscript):
                agg = lhs.aggregate
                lhs = prim.Subscript(agg, prim.Variable(i_outer))
            else:
                raise NotImplementedError(
                    "LHS of a reduction must be a subscript !"
                )

            insns.append(
                lp.Assignment(
                    lhs,
                    expression=rhs,
                    within_inames=frozenset({i_outer}),
                    id=knl.get_var_name_generator()(LOOPY_INSN_PREFIX),
                    predicates=insn.predicates,
                )
            )
        else:
            insns.append(insn)

    tunit = lp.make_kernel(
        knl.domains,
        insns,
        knl.args,
        name=knl.name,
        target=knl.target,
        applied_iname_rewrites=knl.applied_iname_rewrites,
        iname_slab_increments=knl.iname_slab_increments,
        lang_version=LOOPY_LANG_VERSION,
    )

    tunit = lp.tag_inames(tunit, {i_outer: "g.0"})
    tunit = lp.tag_inames(tunit, {i_inner: "l.0"})

    if precompute == 1:
        subst = "tmp_sum"
        tunit = reduction_arg_to_subst_rule(
            tunit, i_inner, subst_rule_name=subst
        )
        tunit = lp.precompute(
            tunit,
            subst,
            i_inner,
            temporary_address_space=lp.AddressSpace.LOCAL,
            default_tag="l.0",
        )
    else:
        tunit = lp.realize_reduction(tunit)
        tunit = lp.add_inames_for_unused_hw_axes(tunit)

    return tunit
