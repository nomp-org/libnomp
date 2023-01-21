import cgen as c
import loopy as lp
from loopy.target.c.compyte.dtypes import (
    DTypeRegistry,
    fill_registry_with_c_types,
)


def get_type(val):
    s = ""
    dtype_reg = DTypeRegistry()
    fill_registry_with_c_types(dtype_reg, True)
    if type(val) == lp.kernel.data.ArrayArg:
        ctype = dtype_reg.dtype_to_ctype(val.dtype)
        s = f"({ctype} *)"
        return s
    elif type(val) == lp.kernel.data.ValueArg:
        ctype = dtype_reg.dtype_to_ctype(val.dtype)
        s = f"*({ctype} *)"
        return s
    return s


def set_args(args):
    l = []
    for i in range(len(args)):
        l.append(c.Value(get_type(args[i]), f"args[{i}]"))
    return l


def create_kernel_fun(knl):
    knl_name = lp.generate_code_v2(knl).device_programs[0].name
    list = set_args(knl.callables_table[knl_name].subkernel.args)
    knl_args = list + [c.Value("", "queue"), c.Value("", "nd_range")]
    ndim = knl.default_entrypoint.get_grid_size_upper_bounds_as_exprs(
        knl.callables_table
    )[0]
    knl_func = c.FunctionDeclaration(c.Value("", knl_name), knl_args)
    args = [
        c.Value("sycl::queue", "queue"),
        c.Value(f"sycl::nd_range<{len(ndim)}>", "nd_range"),
        c.Value("unsigned int", "nargs"),
        c.Value("void", "**args"),
    ]
    func = c.FunctionBody(
        c.FunctionDeclaration(
            c.Value('extern "C" void', f"kernel_function_{len(ndim)}"), args
        ),
        c.Block(([knl_func])),
    )
    return str(func)
