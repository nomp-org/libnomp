import loopy as lp

LOOPY_LANG_VERSION = (2018, 2)


def transform(knl, context):
    backend = context["backend"]
    split_size = 8 if (backend == "ispc") else 32
    knl = lp.split_iname(knl, "i", split_size)
    knl = lp.tag_inames(knl, {"i_outer": "g.0", "i_inner": "l.0", "j": "for"})
    return knl
