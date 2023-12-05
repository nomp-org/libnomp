""" Transform script for nomp-api-400. """
import loopy as lp

LOOPY_LANG_VERSION = (2018, 2)


# pylint: disable=unused-argument
def static(knl, context):
    """Tag `i` as global and `j*` as local."""
    knl = lp.tag_inames(
        knl,
        {
            "i": "g.0",
            "j*": "l.0",
        },
    )
    return knl


# pylint: disable=unused-argument
def dynamic(knl, context):
    """Tag `i` as global and `j*` as local."""
    knl = lp.tag_inames(
        knl,
        {
            "i": "g.0",
            "j*": "l.0",
        },
    )
    knl = lp.fix_parameters(knl, m=16)
    return knl
