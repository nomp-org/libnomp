"""Module to do reductions with using Loopy."""
from dataclasses import dataclass

import loopy as lp
import pymbolic.primitives as prim
from loopy.transform.data import reduction_arg_to_subst_rule
from loopy_api import LOOPY_LANG_VERSION

_BACKEND_TO_BLOCK_SIZE = {"cuda": 32, "opencl": 32}


@dataclass
class ReductionInfo:
    """Store meta information about reductions."""

    var: str = ""
    arg: lp.KernelArgument = None

    @property
    def tmp(self):
        """Return the name of the temporary variable name used for reduction"""
        return f"{self.var}_tmp"


def realize_reduction(
    tunit: lp.translation_unit.TranslationUnit, backend: str
) -> tuple[lp.translation_unit.TranslationUnit, int]:
    """Perform transformations to realize reduction."""

    if len(tunit.callables_table.keys()) > 1:
        raise NotImplementedError(
            "Don't know how to handle more than 1 callable in translation unit!"
        )
    (knl_name,) = tunit.callables_table.keys()

    knl, redn = tunit.default_entrypoint, ReductionInfo()
    for insn in knl.instructions:
        if isinstance(insn, lp.Assignment):
            if isinstance(insn.assignee, prim.Subscript) and isinstance(
                insn.expression, lp.symbolic.Reduction
            ):
                if isinstance(insn.assignee.aggregate, prim.Variable):
                    redn.var = insn.assignee.aggregate.name
    if not redn.var:
        raise SyntaxError("Reduction variable or operation not found.")

    if len(knl.inames.keys()) > 1:
        raise NotImplementedError(
            "Don't know how to handle more than 1 iname in redcution !"
        )

    (iname,) = knl.inames
    i_inner, i_outer = f"{iname}_inner", f"{iname}_outer"
    tunit = lp.split_iname(
        tunit,
        iname,
        _BACKEND_TO_BLOCK_SIZE[backend],
        inner_iname=i_inner,
        outer_iname=i_outer,
    )
    tunit = lp.split_reduction_outward(tunit, i_outer)
    tunit = lp.split_reduction_inward(tunit, i_inner)

    tunit = reduction_arg_to_subst_rule(
        tunit, i_outer, subst_rule_name=redn.tmp
    )
    tunit = lp.precompute(
        tunit,
        redn.tmp,
        i_outer,
        temporary_address_space=lp.AddressSpace.GLOBAL,
        default_tag="l.auto",
    )
    tunit = lp.realize_reduction(tunit)
    knl = tunit[knl_name]

    for arg in knl.args:
        if arg.name == redn.var:
            redn.arg = arg
            break

    tvar = knl.temporary_variables.pop(f"{redn.tmp}_0")
    knl.args.append(
        lp.GlobalArg(
            tvar.name,
            dtype=redn.arg.dtype,
            shape=tvar.shape,
            dim_tags=tvar.dim_tags,
            offset=tvar.offset,
            dim_names=tvar.dim_names,
            alignment=tvar.alignment,
            tags=tvar.tags,
        )
    )
    knl.args.sort(key=lambda arg: arg.name.lower())

    knl = lp.make_kernel(
        knl.domains,
        knl.instructions,
        knl.args + list(knl.temporary_variables.values()),
        name=knl.name,
        target=knl.target,
        lang_version=LOOPY_LANG_VERSION,
    )

    knl = lp.tag_inames(knl, {i_inner: "l.0"})
    knl = lp.tag_inames(knl, {f"{i_outer}_0": "g.0"})
    knl = lp.add_dependency(knl, "writes:acc_i_outer", f"id:{redn.tmp}_barrier")
    return knl
