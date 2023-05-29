import loopy as lp

LOOPY_LANG_VERSION = (2018, 2)


def transform(knl, context):
    def split_and_tag(knl, i, axis):
        i_inner, i_outer = f"{i}_inner", f"{i}_outer"
        knl = lp.split_iname(
            knl, i, 32, inner_iname=i_inner, outer_iname=i_outer
        )
        knl = lp.tag_inames(knl, {i_outer: f"g.{axis}", i_inner: f"l.{axis}"})
        return knl

    (i, j) = knl.default_entrypoint.all_inames()
    knl = split_and_tag(knl, i, 0)
    knl = split_and_tag(knl, j, 1)

    return knl
