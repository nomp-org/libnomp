import loopy as lp


def realize_reduction(
    knl: lp.translation_unit.TranslationUnit, backend: str
) -> lp.translation_unit.TranslationUnit:
    # inames = knl.callables_table["loopy_knl"].subkernel.inames
    # knl = lp.split_iname(
    #     knl, "i", 32, inner_iname="i_inner", outer_iname="i_outer_0"
    # )
    # knl = lp.tag_inames(knl, {"i_inner": "l.0"})
    # knl = lp.split_reduction_outward(knl, "i_outer_0")
    # knl = lp.split_reduction_inward(knl, "i_inner")
    return knl
