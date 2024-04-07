"""Transform script for nomp-api-225, nomp-api-240, nomp-api-300 and
nomp-api-350."""

import loopy as lp

LOOPY_LANG_VERSION = (2018, 2)


def tile_outer(knl, context):
    """Tile outer iname and tag inner iname as sequential."""
    (i, j) = sorted(knl.default_entrypoint.all_inames())
    block_size = min(512, context["device::max_threads_per_block"])
    i_inner, i_outer = f"{i}_inner", f"{i}_outer"
    knl = lp.split_iname(
        knl, i, block_size, inner_iname=i_inner, outer_iname=i_outer
    )
    knl = lp.tag_inames(knl, {i_outer: "g.0", i_inner: "l.0", j: "for"})
    return knl
