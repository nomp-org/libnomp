"""Loopy Kernel wrappers"""
from typing import Callable

import cgen as c
import loopy as lp
from loopy.target.c.compyte.dtypes import DTypeRegistry


class BaseKernelWrapper:
    """Base loopy kernel wrapper"""

    def __init__(self, prefix=None, includes=None):
        self.reg = DTypeRegistry()
        self.prefix = prefix
        self.includes = includes

    def get_src(self, knl: lp.translation_unit.TranslationUnit) -> str:
        """Get kernel source"""
        return lp.generate_code_v2(knl).device_code()

    def get_entry_point(self, knl: lp.translation_unit.TranslationUnit) -> str:
        """Get wrapper name"""
        entry = knl.default_entrypoint.name
        if self.prefix:
            return f"{self.prefix}_{entry}"
        return entry

    def get_device_code_and_args(
        self,
        knl: lp.translation_unit.TranslationUnit,
        get_arg: Callable[[DTypeRegistry, int, str], c.Value],
    ) -> tuple[str, str, str, list]:
        """Get device code and arguments"""
        gen_code = lp.generate_code_v2(knl)
        device_code = gen_code.device_code()
        knl_name = gen_code.device_programs[0].name
        entry = knl.default_entrypoint.name
        args_list = [
            get_arg(self.reg, index, value)
            for index, value in enumerate(knl[entry].args)
            if not None
        ]
        wrapper_name = self.get_entry_point(knl)
        return knl_name, wrapper_name, device_code, args_list
