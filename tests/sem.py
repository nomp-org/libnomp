"""Kernel Annotation function"""
from typing import Dict

import loopy as lp

LOOPY_LANG_VERSION = (2018, 2)


def annotate(
    knl: lp.translation_unit.TranslationUnit, annotations: Dict[str, str]
) -> lp.translation_unit.TranslationUnit:
    """Annotate loops using group_ids and local_ids"""
    inames = knl.default_entrypoint.all_inames()
    dof_axis = 0
    for key in annotations:
        if key == "dof_loop":
            loop = annotations[key]
            if loop in inames:
                knl = lp.tag_inames(knl, [(loop, f"l.{dof_axis}")])
                dof_axis += 1
        elif key == "element_loop":
            loop = annotations[key]
            if loop in inames:
                knl = lp.tag_inames(knl, [(loop, "g.0")])
        elif key == "grid_loop":
            loop = annotations[key]
            if loop in inames:
                knl = lp.split_iname(knl, loop, 32)
                knl = lp.tag_inames(
                    knl, [(f"{loop}_outer", "g.0"), (f"{loop}_inner", "l.0")]
                )
    return knl
