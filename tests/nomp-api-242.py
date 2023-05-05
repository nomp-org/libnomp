import loopy as lp

LOOPY_LANG_VERSION = (2018, 2)


def transform(knl, context):
    (g,) = knl.default_entrypoint.all_inames()
    knl = lp.tag_inames(knl, [(g, "g.0")])
    return knl
