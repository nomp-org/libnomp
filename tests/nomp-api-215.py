import loopy as lp

LOOPY_LANG_VERSION = (2018, 2)


def transform(knl, context):
    knl = lp.split_iname(
        knl, "i", 32, inner_iname="i_inner", outer_iname="i_outer"
    )
    knl = lp.tag_inames(knl, {"i_outer": "g.0", "i_inner": "l.0", "j": "for"})
    return knl
