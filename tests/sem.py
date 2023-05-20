from typing import Dict

import loopy as lp

LOOPY_LANG_VERSION = (2018, 2)


def annotate(
    knl: lp.translation_unit.TranslationUnit,
    annotations: Dict[str, str],
    context: Dict[str, str],
) -> lp.translation_unit.TranslationUnit:
    inames = knl.default_entrypoint.all_inames()
    dof_axis = 0
    backend = context["backend"]
    for key in annotations:
        if key == "dof_loop":
            loop = annotations[key]
            if loop in inames:
                split_size = 8 if (backend == "ispc") else 32
                lp.split_iname(knl, loop, split_size, inner_tag=f"l.{dof_axis}")
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
