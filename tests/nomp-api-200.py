import loopy as lp

LOOPY_LANG_VERSION = (2018, 2)


def transform(knl):
    (g,) = knl.default_entrypoint.all_inames()
    knl = lp.tag_inames(knl, [(g, "g.0")])
    return knl


def matrix_transform(knl):
    (g0, g1) = knl.default_entrypoint.all_inames()
    knl = lp.tag_inames(knl, [(g0, "g.0"), (g1, "g.1")])
    return knl
