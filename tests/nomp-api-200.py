import loopy as lp

LOOPY_LANG_VERSION = (2018, 2)


def foo_old(knl):
    (g,) = knl.default_entrypoint.all_inames()
    knl = lp.tag_inames(knl, [(g, "g.0")])
    return knl


def foo(knl):
    (iname,) = knl.default_entrypoint.all_inames()
    knl = lp.split_iname(knl, iname, 32)
    knl = lp.tag_inames(
        knl, [(f"{iname}_outer", "g.0"), (f"{iname}_inner", "l.0")]
    )
    return knl
