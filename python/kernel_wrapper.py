import cgen as c
import loopy as lp
from loopy.target.c.compyte.dtypes import (
    DTypeRegistry,
    fill_registry_with_c_types,
)


def get_arg_type(val,dtype_reg):
    arg_type = "" 
    fill_registry_with_c_types(dtype_reg, True)
    if type(val) == lp.kernel.data.ArrayArg:
        ctype = dtype_reg.dtype_to_ctype(val.dtype)
        arg_type = f"({ctype} *)"
        return arg_type
    elif type(val) == lp.kernel.data.ValueArg:
        ctype = dtype_reg.dtype_to_ctype(val.dtype)
        arg_type = f"*({ctype} *)"
        return arg_type
    return arg_type

def create_kernel_wrapper_fun(knl):
    """Create SYCL kernel wrapper"""
    dtype_reg=DTypeRegistry()
    knl_name = lp.generate_code_v2(knl).device_programs[0].name
    args = knl.callables_table[knl_name].subkernel.args
    args_list = [
        c.Value(get_arg_type(value,dtype_reg), f"args[{index}]")
        for index, value in enumerate(args)
    ]
    args_length = len(args)
    grid_size = knl.default_entrypoint.get_grid_size_upper_bounds_as_exprs(
        knl.callables_table
    )
    py_global = grid_size[0]
    py_local = grid_size[1]
    ndim = len(py_global)
    if len(py_global) < len(py_local):
        ndim = len(py_local)
    knl_args = args_list + [
        c.Value("*(sycl::queue *)", f"args[{args_length}]"),
        c.Value(f"*(sycl::nd_range<{ndim}> *)", f"args[{args_length+1}]"),
    ]
    knl_function_call = c.FunctionDeclaration(c.Value("", knl_name), knl_args)
    function_args = [
        c.Value("void", "**args"),
    ]
    function_body = c.FunctionBody(
        c.FunctionDeclaration(
            c.Value('extern "C" void', "kernel_function"), function_args
        ),
        c.Block(([knl_function_call])),
    )
    return str(function_body)