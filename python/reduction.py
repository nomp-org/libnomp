"""Module to do reductions with using Loopy."""
import loopy as lp
import pymbolic.primitives as prim
from loopy.transform.data import reduction_arg_to_subst_rule
from loopy_api import LOOPY_LANG_VERSION

_BACKEND_TO_BLOCK_SIZE = {"cuda": 32, "opencl": 32}


def realize_reduction(
    tunit: lp.translation_unit.TranslationUnit, backend: str
) -> lp.translation_unit.TranslationUnit:
    """Perform transformations to realize reduction."""

    if len(tunit.callables_table.keys()) > 1:
        raise NotImplementedError(
            "Don't know how to handle more than 1 callable in TU!"
        )
    (knl_name,) = tunit.callables_table.keys()

    redn_var = None
    for insn in tunit.default_entrypoint.instructions:
        if isinstance(insn, lp.Assignment):
            if isinstance(insn.assignee, prim.Subscript) and isinstance(
                insn.expression, lp.symbolic.Reduction
            ):
                if isinstance(insn.assignee.aggregate, prim.Variable):
                    redn_var = insn.assignee.aggregate.name
    if redn_var is None:
        return tunit

    if len(tunit.default_entrypoint.inames.keys()) > 1:
        raise NotImplementedError(
            "Don't know how to handle more than 1 iname in redcution !"
        )

    (iname,) = tunit.default_entrypoint.inames
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

    redn_tmp = f"{redn_var}_tmp"
    tunit = reduction_arg_to_subst_rule(
        tunit, i_outer, subst_rule_name=redn_tmp
    )
    tunit = lp.precompute(
        tunit,
        redn_tmp,
        i_outer,
        temporary_address_space=lp.AddressSpace.GLOBAL,
        default_tag="l.auto",
    )
    tunit = lp.realize_reduction(tunit)

    knl = tunit[knl_name]
    args, tmp_vars, redn_arg = knl.args, knl.temporary_variables, None
    for arg in args:
        if arg.name == redn_var:
            redn_arg = arg
            break

    tmp_var = tmp_vars.pop(f"{redn_tmp}_0")
    args.append(
        lp.GlobalArg(
            tmp_var.name,
            dtype=redn_arg.dtype,
            shape=tmp_var.shape,
            dim_tags=tmp_var.dim_tags,
            offset=tmp_var.offset,
            dim_names=tmp_var.dim_names,
            alignment=tmp_var.alignment,
            tags=tmp_var.tags,
        )
    )
    args.sort(key=lambda arg: arg.name.lower())

    tunit = lp.make_kernel(
        knl.domains,
        knl.instructions,
        args + list(tmp_vars.values()),
        lang_version=LOOPY_LANG_VERSION,
    )

    tunit = lp.tag_inames(tunit, {i_inner: "l.0"})
    tunit = lp.tag_inames(tunit, {f"{i_outer}_0": "g.0"})
    return tunit
