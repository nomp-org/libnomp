"""Module to do reductions with using Loopy."""

import loopy as lp
import pymbolic.primitives as prim
from loopy.symbolic import Reduction
from loopy.transform.data import reduction_arg_to_subst_rule
from loopy_api import LOOPY_LANG_VERSION

_TARGET_BLOCK_SIZE = {"cuda": 32, "opencl": 32}


def target_to_str(target: lp.target.TargetBase) -> str:
    """Get the target name as a string"""
    if isinstance(target, lp.target.cuda.CudaTarget):
        return "cuda"
    if isinstance(target, lp.target.opencl.OpenCLTarget):
        return "opencl"
    if isinstance(target, lp.target.ispc.ISPCTarget):
        return "ispc"
    raise NotImplementedError(f"Uknown target: {target}")


def realize_reduction(
    tunit: lp.translation_unit.TranslationUnit, var: str
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
        _TARGET_BLOCK_SIZE[target_to_str(knl.target)],
        inner_iname=i_inner,
        outer_iname=i_outer,
    )

    (knl_name,) = tunit.callables_table.keys()
    knl = tunit[knl_name]

    insns = []
    for insn in knl.instructions:
        if (
            isinstance(insn, lp.Assignment)
            and isinstance(insn.assignee, prim.Subscript)
            and isinstance(insn.assignee.aggregate, prim.Variable)
            and insn.assignee.aggregate.name == var
        ):
            (lhs, rhs) = insn.expression.children
            if isinstance(insn.expression, prim.Sum):
                rhs = Reduction(
                    lp.library.reduction.SumReductionOperation(),
                    i_inner,
                    rhs,
                )
            elif isinstance(insn.expression, prim.Product):
                rhs = Reduction(
                    lp.library.reduction.ProductReductionOperation(),
                    i_inner,
                    rhs,
                )

            if isinstance(lhs, prim.Subscript):
                agg = lhs.aggregate
                lhs = prim.Subscript(agg, prim.Variable(i_outer))
            else:
                raise NotImplementedError(
                    "LHS of a reduction must be a subscript !"
                )
            # FIXME: Missing predicates. `id` has to be random and prefixed
            # with "nomp_insn_".
            insn_id = "sum_reduction_"
            insns.append(
                lp.Assignment(
                    lhs,
                    expression=rhs,
                    within_inames=frozenset({i_outer}),
                    id=insn_id,
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
        lang_version=LOOPY_LANG_VERSION,
    )

    subst = "tmp_sum"
    tunit = reduction_arg_to_subst_rule(tunit, i_inner, subst_rule_name=subst)
    tunit = lp.precompute(
        tunit,
        subst,
        i_inner,
        temporary_address_space=lp.AddressSpace.LOCAL,
        default_tag="l.0",
    )

    tunit = lp.tag_inames(tunit, {i_outer: "g.0"})
    tunit = lp.tag_inames(tunit, {i_inner: "l.0"})
    tunit = lp.remove_unused_inames(tunit)

    return tunit
