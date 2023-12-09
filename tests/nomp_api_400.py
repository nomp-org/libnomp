""" Transform script for nomp-api-400. """
import loopy as lp

LOOPY_LANG_VERSION = (2018, 2)


# pylint: disable=unused-argument
def transform(knl, context):
    """Tag `i` as global and `j*` as local."""
    knl = lp.tag_inames(
        knl,
        {
            "i": "g.0",
            "j*": "l.0",
        },
    )
    return knl
