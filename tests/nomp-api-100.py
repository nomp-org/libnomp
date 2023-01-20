import loopy as lp

LOOPY_LANG_VERSION = (2018, 2)


def transform(knl):
    (g,) = knl.default_entrypoint.all_inames()
    knl = lp.tag_inames(knl, [(g, "g.0")])
    return knl


def invalid_func(knl):
    (g,) = knl.default_entrypoint.all_names()
    knl = lp.tag_inames(knl, [(g, "g.0")])
    return kn  # noqa
