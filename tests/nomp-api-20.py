import loopy as lp

LOOPY_LANG_VERSION = (2018, 2)


def transform(knl):
    inames = knl.default_entrypoint.all_inames()
    if len(inames) == 1:
        (g,) = inames
        knl = lp.tag_inames(knl, [(g, "g.0")])
    elif len(inames) == 2:
        (g, l) = inames
        (g, l) = sorted((g, l))
        knl = lp.tag_inames(knl, [(g, "g.0"), (l, "l.0")])
    elif len(inames) == 3:
        (g, l, s) = inames
        (g, l, s) = sorted((g, l, s))
        knl = lp.tag_inames(knl, [(g, "g.0"), (l, "l.0"), (s, "for")])
    return knl
