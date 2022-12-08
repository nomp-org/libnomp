"""Transformation functions for tests nomp-api-52 and nomp-api-53"""
import loopy as lp

LOOPY_LANG_VERSION = (2018, 2)


def transform(knl):
    """Map iname to group_id"""
    (gid_0,) = knl.default_entrypoint.all_inames()
    knl = lp.tag_inames(knl, [(gid_0, "g.0")])
    return knl


def invalid_func(knl):
    """Transform function with a runtime error"""
    (gid_0,) = knl.default_entrypoint.all_names()
    knl = lp.tag_inames(knl, [(gid_0, "g.0")])
    return kn  # noqa pylint: disable=E0602
