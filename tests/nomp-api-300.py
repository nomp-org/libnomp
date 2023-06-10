import loopy as lp

LOOPY_LANG_VERSION = (2018, 2)


def transform(knl, context):
    backend = context["backend"]
    def split_and_tag(knl, i, axis):
        split_size = 8 if (backend == "ispc") else 32
        i_inner, i_outer = f"{i}_inner", f"{i}_outer"
        knl = lp.split_iname(
            knl, i, split_size, inner_iname=i_inner, outer_iname=i_outer
        )
        if (backend == "ispc" and axis >= 1):
            knl = lp.tag_inames(knl, {i_outer: f"g.{axis}"})
        else:
            knl = lp.tag_inames(knl, {i_outer: f"g.{axis}", i_inner: f"l.{axis}"})
        return knl

    (i, j) = knl.default_entrypoint.all_inames()
    knl = split_and_tag(knl, i, 0)
    knl = split_and_tag(knl, j, 1)

    return knl


def mxm_transform(knl, context):
    backend = context["backend"]
    split_size = 8 if (backend == "ispc") else 32
    knl = lp.split_iname(knl, "i", split_size)
    knl = lp.split_iname(knl, "j", split_size)
    iname_to_tag = {
            "i_outer": "g.0",
            "i_inner": "l.0",
            "j_outer": "g.1",
            "k": "for",
        }
    if (backend != "ispc"): iname_to_tag["j_inner"] = "l.1"
    knl = lp.tag_inames(
        knl,
        iname_to_tag,
    )
    return knl


def vxm_transform(knl, context):
    backend = context["backend"]
    split_size = 8 if (backend == "ispc") else 32
    knl = lp.split_iname(knl, "i", split_size)
    knl = lp.tag_inames(knl, {"i_outer": "g.0", "i_inner": "l.0", "j": "for"})
    return knl
