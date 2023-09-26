import loopy as lp

LOOPY_LANG_VERSION = (2018, 2)


def transform(knl, context):
    block_size = min(512, context["device::max_threads_per_block"])
    knl = lp.split_iname(knl, "i", block_size)
    knl = lp.tag_inames(knl, {"i_outer": "g.0", "i_inner": "l.0", "j": "for"})
    return knl
