import loopy as lp

LOOPY_LANG_VERSION = (2018, 2)


def transform(knl, context):
    knl = lp.tag_inames(knl, [("i", "g.0"), ("j", "for")])
    return knl
