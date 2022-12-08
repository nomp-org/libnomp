"""Transformation functions for test nomp-api-300"""
import loopy as lp

LOOPY_LANG_VERSION = (2018, 2)


def transform(knl):
    """Map inames to group_ids"""
    (gid_0, gid_1) = knl.default_entrypoint.all_inames()
    knl = lp.tag_inames(knl, [(gid_0, "g.0"), (gid_1, "g.1")])
    return knl
