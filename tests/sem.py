"""Annotation script for the spectral element kernels."""

from typing import Dict

import loopy as lp

LOOPY_LANG_VERSION = (2018, 2)


def annotate(
    knl: lp.translation_unit.TranslationUnit,
    annotations: Dict[str, str],
    context: Dict[str, str],
) -> lp.translation_unit.TranslationUnit:
    """Annotate the spectral element kernels based on domain knowledge."""
    inames = knl.default_entrypoint.all_inames()
    block_size = min(512, context["device::max_threads_per_block"])
    dof_axis = 0
    for key in annotations:
        if key == "dof_loop":
            loop = annotations[key]
            if loop in inames:
                lp.split_iname(knl, loop, block_size, inner_tag=f"l.{dof_axis}")
                dof_axis += 1
        if key == "element_loop":
            loop = annotations[key]
            if loop in inames:
                knl = lp.tag_inames(knl, [(loop, "g.0")])
        if key == "grid_loop":
            loop = annotations[key]
            if loop in inames:
                knl = lp.split_iname(knl, loop, block_size)
                knl = lp.tag_inames(
                    knl, [(f"{loop}_outer", "g.0"), (f"{loop}_inner", "l.0")]
                )
    return knl
