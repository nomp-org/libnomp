import loopy as lp
from loopy.transform.data import reduction_arg_to_subst_rule


def realize_reduction(
    knl: lp.translation_unit.TranslationUnit, backend: str, redn_var: str
) -> lp.translation_unit.TranslationUnit:
    for insn in knl.default_entrypoint.instructions:
        if (
            isinstance(insn, lp.kernel.instruction.Assignment)
            and insn.assignee_name == redn_var
        ):
            if len(insn.expression.inames) != 1:
                raise SyntaxError("Don't know how to handle this reduction !")

            (redn_iname,) = insn.expression.inames
            i_inner, i_outer = f"{redn_iname}_inner", f"{redn_iname}_outer"
            knl = lp.split_iname(
                knl, redn_iname, 32, inner_iname=i_inner, outer_iname=i_outer
            )
            knl = lp.tag_inames(knl, {i_inner: "l.0"})
            knl = lp.split_reduction_outward(knl, i_outer)
            knl = lp.split_reduction_inward(knl, i_inner)

            knl = reduction_arg_to_subst_rule(knl, i_outer)

            knl = lp.precompute(
                knl,
                f"red_{i_outer}_arg",
                i_outer,
                temporary_address_space=lp.AddressSpace.GLOBAL,
                default_tag="l.auto",
            )

            knl = lp.realize_reduction(knl)

            knl = lp.tag_inames(knl, {f"{i_outer}_0": "g.0"})

            knl = lp.add_dependency(
                knl,
                f"writes:acc_{i_outer}",
                f"id:red_{i_outer}_arg_barrier",
            )
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
