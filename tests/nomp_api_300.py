""" Transform script for nomp-api-300. """
import math

import loopy as lp

LOOPY_LANG_VERSION = (2018, 2)


def madd_transform(knl, context):
    """Tile both the loops."""
    block_size = int(
        math.sqrt(min(1024, context["device::max_threads_per_block"]))
    )
    knl = lp.split_iname(knl, "i", block_size)
    knl = lp.split_iname(knl, "j", block_size)

    knl = lp.tag_inames(
        knl,
        {
            "i_outer": "g.0",
            "i_inner": "l.0",
            "j_outer": "g.1",
            "j_inner": "l.1",
        },
    )
    return knl


def mxm_transform(knl, context):
    """Tile two outer loops and tag innermost loop as sequential."""
    block_size = int(
        math.sqrt(min(1024, context["device::max_threads_per_block"]))
    )
    knl = lp.split_iname(knl, "i", block_size)
    knl = lp.split_iname(knl, "j", block_size)

    knl = lp.tag_inames(
        knl,
        {
            "i_outer": "g.0",
            "i_inner": "l.0",
            "j_outer": "g.1",
            "j_inner": "l.1",
            "k": "for",
        },
    )
    return knl
