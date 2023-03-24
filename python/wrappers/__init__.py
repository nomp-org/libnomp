import cgen as c
import loopy as lp
from loopy.target.c.compyte.dtypes import (
    DTypeRegistry,
    fill_registry_with_c_types,
)
from loopy.target.ispc import fill_registry_with_ispc_types


class BaseWrapper:
    """Base loopy kernel wrapper"""

    def get_entry_point(self, knl: lp.translation_unit.TranslationUnit) -> str:
        """Get entry point"""
        (entry,) = knl.entrypoints
        return entry

    def get_src(self, knl: lp.translation_unit.TranslationUnit) -> str:
        return lp.generate_code_v2(knl).device_code()


class ISPCWrapper(BaseWrapper):
    """ISPC loopy kernel wrapper"""

    def get_entry_point(self, knl: lp.translation_unit.TranslationUnit) -> str:
        """Get ISPC entry point"""
        gen_code = lp.generate_code_v2(knl)
        knl_name = gen_code.device_programs[0].name
        return f"nomp_{knl_name}_wrapper"

    def get_src(self, knl: lp.translation_unit.TranslationUnit) -> str:
        """Create ISPC kernel wrapper"""

        def get_arg(reg, index, value):
            """Get ISPC type"""
            ctype = reg.dtype_to_ctype(value.dtype)
            if isinstance(value, lp.kernel.data.ArrayArg):
                return f"({ctype} *uniform)(p[{index}])"
            if isinstance(value, lp.kernel.data.ValueArg):
                return f"*(({ctype} *)(p[{index}]))"
            return ""

        reg = DTypeRegistry()
        fill_registry_with_ispc_types(reg, True)
        gen_code = lp.generate_code_v2(knl)
        device_code = gen_code.device_code()
        knl_name = gen_code.device_programs[0].name
        wrapper_name = self.get_entry_point(knl)
        (entry,) = knl.entrypoints
        knl_args = [
            c.Value("", get_arg(reg, index, value))
            for index, value in enumerate(knl[entry].args)
        ]
        param_count = len(knl_args)
        wrapper = c.FunctionBody(
            c.FunctionDeclaration(
                c.Value("task void", wrapper_name),
                [c.Value("void *uniform", "_p")],
            ),
            c.Block(
                (
                    [c.Statement("void **uniform p = (void **uniform)_p")]
                    + [
                        c.Statement(
                            f"uniform int dim{i} = *((int"
                            f" *)(p[{param_count + i}]))"
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
        entry_point = c.Line(
            f"""#include "ispcrt.isph"\nDEFINE_CPU_ENTRY_POINT({wrapper_name})"""
        )
        content = list(map(str, [device_code, wrapper, entry_point]))

        return "\n\n".join(content).strip()


class SyclWrapper(BaseWrapper):
    """Sycl loopy kernel wrapper"""

    def get_entry_point(self, knl: lp.translation_unit.TranslationUnit) -> str:
        """Get Sycl entry point"""
        return "kernel_function"

    def get_src(self, knl: lp.translation_unit.TranslationUnit) -> str:
        """Create SYCL kernel wrapper"""

        def get_arg_type(val, dtype_reg):
            arg_type = ""
            if type(val) == lp.kernel.data.ArrayArg:
                ctype = dtype_reg.dtype_to_ctype(val.dtype)
                arg_type = f"({ctype} *)"
                return arg_type
            elif type(val) == lp.kernel.data.ValueArg:
                ctype = dtype_reg.dtype_to_ctype(val.dtype)
                arg_type = f"*({ctype} *)"
                return arg_type
            return arg_type

        dtype_reg = DTypeRegistry()
        fill_registry_with_c_types(dtype_reg, True)
        knl_name = lp.generate_code_v2(knl).device_programs[0].name
        args = knl.callables_table[knl_name].subkernel.args
        args_list = [
            c.Value(get_arg_type(value, dtype_reg), f"args[{index}]")
            for index, value in enumerate(args)
        ]
        args_length = len(args_list)
        grid_size = knl.default_entrypoint.get_grid_size_upper_bounds_as_exprs(
            knl.callables_table
        )
        py_global, py_local, _ = grid_size
        ndim = max(len(py_global), len(py_local))
        knl_args = args_list + [
            c.Value("*(sycl::queue *)", f"args[{args_length}]"),
            c.Value(f"*(sycl::nd_range<{ndim}> *)", f"args[{args_length + 1}]"),
        ]
        wrapper_name = self.get_entry_point(knl)
        function_body = c.FunctionBody(
            c.FunctionDeclaration(
                c.Value('extern "C" void', wrapper_name),
                [c.Value("void", "**args")],
            ),
            c.Block(([c.FunctionDeclaration(c.Value("", knl_name), knl_args)])),
        )

        return str(function_body)
