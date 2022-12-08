"""Transformation functions for tests nomp-api-2**"""
import loopy as lp

LOOPY_LANG_VERSION = (2018, 2)


def transform(knl):
    """Map iname to group_id"""
    (gid_0,) = knl.default_entrypoint.all_inames()
    knl = lp.tag_inames(knl, [(gid_0, "g.0")])
    return knl
