"""Loopy Kernel wrappers"""
from typing import Callable

import cgen as c
import loopy as lp
from loopy.target.c.compyte.dtypes import (
    DTypeRegistry,
    fill_registry_with_c_types,
)
from loopy.target.ispc import fill_registry_with_ispc_types


def combine(structs: list) -> str:
    """Combine list to single string"""
    content = list(map(str, structs))
    return "\n\n".join(content).strip()


class BaseKernelWrapper:
    """Base loopy kernel wrapper"""

    def __init__(self, prefix=None, includes=None):
        self.reg = DTypeRegistry()
        self.prefix = prefix
        self.includes = includes

    def get_src(
        self, knl: lp.translation_unit.TranslationUnit, redn_arg: str
    ) -> str:
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
        redn_arg: str,
    ) -> tuple[str, str, str, list]:
        """Get device code and arguments"""
        gen_code = lp.generate_code_v2(knl)
        device_code = gen_code.device_code()
        knl_name = gen_code.device_programs[0].name
        entry = knl.default_entrypoint.name
        redn_args_count = 0
        args_list = []
        for index, value in enumerate(knl[entry].args):
            if value.name != redn_arg:
                args_list.append(
                    get_arg(self.reg, index - redn_args_count, value)
                )
            else:
                redn_args_count += 1

        wrapper_name = self.get_entry_point(knl)
        return knl_name, wrapper_name, device_code, args_list


class ISPCKernelWrapper(BaseKernelWrapper):
    """ISPC loopy kernel wrapper"""

    def __init__(self, prefix=None, includes=None):
        super().__init__(prefix, includes)
        fill_registry_with_ispc_types(self.reg, True)

    def get_src(
        self, knl: lp.translation_unit.TranslationUnit, redn_arg: str
    ) -> str:
        """Create ISPC kernel wrapper"""

        def get_arg(reg, index, value):
            """Get ISPC type"""
            ctype = reg.dtype_to_ctype(value.dtype)
            if isinstance(value, lp.kernel.data.ArrayArg):
                return c.Value("", f"({ctype} *uniform)(p[{index}])")
            if isinstance(value, lp.kernel.data.ValueArg):
                return c.Value("", f"*(({ctype} *)(p[{index}]))")
            return None

        (
            knl_name,
            wrapper_name,
            device_code,
            knl_args,
        ) = self.get_device_code_and_args(knl, get_arg, redn_arg)
        wrapper = c.FunctionBody(
            c.FunctionDeclaration(
                c.Value("export void", wrapper_name),
                [c.Value("void *uniform", "_p")],
            ),
            c.Block(
                (
                    [c.Statement("void **uniform p = (void **uniform)_p")]
                    + [
                        c.Statement(
                            f"uniform int dim{i} = *((int"
                            f" *)(p[{len(knl_args) + i}]))"
                        )
                        for i in range(3)
                    ]
                    + [
                        c.FunctionDeclaration(
                            c.Value("launch[dim0, dim1, dim2]", knl_name),
                            knl_args,
                        ),
                    ]
                )
            ),
        )
        return combine([device_code, wrapper])


class SyclKernelWrapper(BaseKernelWrapper):
    """Sycl loopy kernel wrapper"""

    def __init__(self, prefix=None, includes=None):
        super().__init__(prefix, includes)
        fill_registry_with_c_types(self.reg, True)

    def get_src(
        self, knl: lp.translation_unit.TranslationUnit, redn_arg: str
    ) -> str:
        """Create SYCL kernel wrapper"""

        def get_arg(reg, index, value):
            """Get SYCL type"""
            ctype = reg.dtype_to_ctype(value.dtype)
            if isinstance(value, lp.kernel.data.ArrayArg):
                return c.Value(f"({ctype} *)", f"args[{index}]")
            if isinstance(value, lp.kernel.data.ValueArg):
                return c.Value(f"*({ctype} *)", f"args[{index}]")
            return None

        (
            knl_name,
            wrapper_name,
            device_code,
            args_list,
        ) = self.get_device_code_and_args(knl, get_arg, redn_arg)
        knl_args = args_list + [
            c.Value("*(sycl::queue *)", f"args[{len(args_list)}]"),
            c.Value("*(sycl::nd_range<3> *)", f"args[{len(args_list) + 1}]"),
        ]
        function_body = c.FunctionBody(
            c.FunctionDeclaration(
                c.Value('extern "C" void', wrapper_name),
                [c.Value("void", "**args")],
            ),
            c.Block(([c.FunctionDeclaration(c.Value("", knl_name), knl_args)])),
        )
        headers = c.Include(self.includes)
        return combine([headers, device_code, function_body])
