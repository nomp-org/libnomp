import loopy as lp
from loopy.transform.data import reduction_arg_to_subst_rule

_BACKEND_TO_TARGET = {"opencl": lp.OpenCLTarget(), "cuda": lp.CudaTarget()}


def realize_reduction(
    knl: lp.translation_unit.TranslationUnit, backend: str, redn_var: str
) -> lp.translation_unit.TranslationUnit:
    if redn_var == "":
        return knl

    if len(knl.callables_table.keys()) > 1:
        raise NotImplementedError(
            "Don't know how to handle more than 1 callable in TU!"
        )
    (knl_name,) = knl.callables_table.keys()

    if len(knl.default_entrypoint.inames.keys()) > 1:
        raise NotImplementedError(
            "Don't know how to handle more than 1 iname in redcution !"
        )

    (iname,) = knl.default_entrypoint.inames
    i_inner, i_outer = f"{iname}_inner", f"{iname}_outer"
    knl = lp.split_iname(
        knl, iname, 32, inner_iname=i_inner, outer_iname=i_outer
    )
    knl = lp.split_reduction_outward(knl, i_outer)
    knl = lp.split_reduction_inward(knl, i_inner)

    redn_tmp = f"{redn_var}_tmp"
    knl = reduction_arg_to_subst_rule(knl, i_outer, subst_rule_name=redn_tmp)
    knl = lp.precompute(
        knl,
        redn_tmp,
        i_outer,
        temporary_address_space=lp.AddressSpace.GLOBAL,
        default_tag="l.auto",
    )
    knl = lp.realize_reduction(knl)

    p = knl[knl_name]
    args, tmp_vars, redn_arg = p.args, p.temporary_variables, None
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

    knl = lp.make_kernel(
        p.domains, p.instructions, args + list(tmp_vars.values())
    )

    knl = lp.tag_inames(knl, {i_inner: "l.0"})
    knl = lp.tag_inames(knl, {f"{i_outer}_0": "g.0"})
    return knl


def test_global_parallel_reduction(knl):
    gsize = 128
    knl = lp.split_iname(knl, "i", gsize * 20)
    knl = lp.split_iname(knl, "i_inner", gsize, inner_tag="l.0")
    knl = lp.split_reduction_outward(knl, "i_outer")
    knl = lp.split_reduction_inward(knl, "i_inner_outer")

    knl = reduction_arg_to_subst_rule(knl, "i_outer")

    knl = lp.precompute(
        knl,
        "red_i_outer_arg",
        "i_outer",
        temporary_address_space=lp.AddressSpace.GLOBAL,
        default_tag="l.auto",
    )
    knl = lp.realize_reduction(knl)
    knl = lp.tag_inames(knl, "i_outer_0:g.0")

    # Keep the i_outer accumulator on the  correct (lower) side of the barrier,
    # otherwise there will be useless save/reload code generated.
    knl = lp.add_dependency(
        knl, "writes:acc_i_outer", "id:red_i_outer_arg_barrier"
    )

    return knl
