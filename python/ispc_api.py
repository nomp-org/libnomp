"""ISPC loopy kernel wrapper"""

import loopy as lp
import numpy as np

from loopy.target.c.compyte.dtypes import DTypeRegistry
import cgen as c


def fill_registry_with_ispc_types(reg):
    """Add ispc types to the registry"""
    reg.get_or_register_dtype("bool", bool)

    reg.get_or_register_dtype(["int8", "signed char", "char"], np.int8)
    reg.get_or_register_dtype(["uint8", "unsigned char"], np.uint8)
    reg.get_or_register_dtype(
        [
            "int16",
            "short",
            "signed short",
            "signed short int",
            "short signed int",
        ],
        np.int16,
    )
    reg.get_or_register_dtype(
        [
            "uint16",
            "unsigned short",
            "unsigned short int",
            "short unsigned int",
        ],
        np.uint16,
    )
    reg.get_or_register_dtype(["int32", "int", "signed int"], np.int32)
    reg.get_or_register_dtype(["uint32", "unsigned", "unsigned int"], np.uint32)

    reg.get_or_register_dtype(["int64"], np.int64)
    reg.get_or_register_dtype(["uint64"], np.uint64)

    reg.get_or_register_dtype("float", np.float32)
    reg.get_or_register_dtype("double", np.float64)


def get_arg(reg, index, value):
    """Get ISPC type"""
    ctype = reg.dtype_to_ctype(value.dtype)
    if isinstance(value, lp.kernel.data.ArrayArg):
        return f"({ctype} *uniform)(p[{index}])"
    if isinstance(value, lp.kernel.data.ValueArg):
        return f"*(({ctype} *)(p[{index}]))"
    return ""


def get_ispc_entry_point(knl):
    """Get ISPC entry point"""
    gen_code = lp.generate_code_v2(knl)
    knl_name = gen_code.device_programs[0].name
    return f"nomp_{knl_name}_wrapper"


def create_ispc_kernel_with_wrapper(knl):
    """Create ISPC kernel wrapper"""
    reg = DTypeRegistry()
    fill_registry_with_ispc_types(reg)
    (entry,) = knl.entrypoints
    knl_args = [
        c.Value("", get_arg(reg, index, value))
        for index, value in enumerate(knl[entry].args)
    ]
    gen_code = lp.generate_code_v2(knl)
    device_code = gen_code.device_code()
    knl_name = gen_code.device_programs[0].name
    wrapper_name = get_ispc_entry_point(knl)
    param_count = len(knl_args)
    calling_func = c.FunctionBody(
        c.FunctionDeclaration(
            c.Value("task void", wrapper_name),
            [
                c.Value("void *uniform", "_p"),
            ],
        ),
        c.Block(
            (
                [
                    c.Statement("void **uniform p = (void **uniform)_p"),
                    c.Statement(
                        f"uniform int dim0 = *((int *)(p[{param_count}]))"
                    ),
                    c.Statement(
                        f"uniform int dim1 = *((int *)(p[{param_count + 1}]))"
                    ),
                    c.Statement(
                        f"uniform int dim2 = *((int *)(p[{param_count + 2}]))"
                    ),
                    c.FunctionDeclaration(
                        c.Value("launch[dim0, dim1, dim2]", knl_name), knl_args
                    ),
                ]
            )
        ),
    )
    entry_point = c.Line(
        f"""#include "ispcrt.isph"\nDEFINE_CPU_ENTRY_POINT({wrapper_name})"""
    )
    content = list(map(str, [device_code, calling_func, entry_point]))
    return "\n\n".join(content).strip()


if __name__ == "__main__":
    import os
    import sys

    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    KNL_STR = """
          void foo(int *a, int N) {
            for (int i = 0; i < N; i++) {
              int t = 0;
              for (int j = 0; j < 10; j++) {
                t += 1;
                if (j == 5)
                  break;
              }
              a[i] = t;
            }
          }
          """
    # lp_knl = c_to_loopy(KNL_STR, "ispc")
    # print(lp.generate_code_v2(lp_knl).device_code())
