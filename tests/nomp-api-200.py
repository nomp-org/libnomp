import loopy as lp

LOOPY_LANG_VERSION = (2018, 2)


def transform(knl, context):
    (iname,) = knl.default_entrypoint.all_inames()
    i_inner, i_outer = f"{iname}_inner", f"{iname}_outer"
    knl = lp.split_iname(
        knl, iname, 32, inner_iname=i_inner, outer_iname=i_outer
    )
    knl = lp.tag_inames(knl, {i_outer: "g.0", i_inner: "l.0"})
    return knl
