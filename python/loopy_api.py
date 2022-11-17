"""Loopy API wrapper"""
from dataclasses import dataclass, fields, replace
from typing import Dict, FrozenSet, List, Optional, Union

import islpy as isl
import loopy as lp
import pymbolic.primitives as prim
from clang import cindex
from loopy.isl_helpers import make_slab
from loopy.kernel.data import AddressSpace
from loopy.symbolic import aff_from_expr
from loopy.target.c.compyte.dtypes import (
    DTypeRegistry,
    fill_registry_with_c_types,
)
from pytools import UniqueNameGenerator, memoize

LOOPY_LANG_VERSION = (2018, 2)
LOOPY_INSN_PREFIX = "_nomp_insn"

_C_BIN_OPS_TO_PYMBOLIC_OPS = {
    "*": lambda x, y: prim.Product((x, y)),
    "+": lambda x, y: prim.Sum((x, y)),
    "-": lambda l, r: prim.Sum((l, prim.Product((-1, r)))),
    "/": prim.Quotient,
    "//": prim.FloorDiv,
    "%": prim.Remainder,
    "<": lambda l, r: prim.Comparison(l, "<", r),
    "<=": lambda l, r: prim.Comparison(l, "<=", r),
    ">": lambda l, r: prim.Comparison(l, ">", r),
    ">=": lambda l, r: prim.Comparison(l, ">=", r),
    "==": lambda l, r: prim.Comparison(l, "==", r),
    "!=": lambda l, r: prim.Comparison(l, "!=", r),
}
_OPS = ["*", "+", "-", "/", "//", "%", "<", ">", "<=", ">=", "==", "!="]
_BACKEND_TO_TARGET = {"opencl": lp.OpenCLTarget(), "cuda": lp.CudaTarget()}


class IdentityMapper:
    """AST node mapper"""

    def rec(self, node, *args, **kwargs):
        """Visit a node."""
        class_name = str(node.kind).split(".")[1]
        try:
            mapper_method = getattr(self, "map_" + class_name.lower())
        except AttributeError as exc:
            raise NotImplementedError(
                f"Mapper method for {class_name} is not implemented."
            ) from exc
        return mapper_method(node, *args, **kwargs)

    __call__ = rec

    def map_integer_literal(self, expr: cindex.CursorKind):
        """Maps int variable"""
        (val,) = expr.get_tokens()
        return (
            dtype_to_ctype_registry()
            .get_or_register_dtype(expr.type.kind.spelling.lower())
            .type
        )(val.spelling)


# Memoize is caching the return of a function based on its input parameters
# so we use it here instead of general @cache.
@memoize
def dtype_to_ctype_registry():
    """Retrieve data type with C type"""
    dtype_reg = DTypeRegistry()
    fill_registry_with_c_types(dtype_reg, True)
    return dtype_reg


# @memoize
def _get_dtype_from_decl_type(decl):
    """Retrieve data type from declaration type"""
    if (
        isinstance(decl.kind, cindex.TypeKind)
        and decl.kind == cindex.TypeKind.POINTER
        and isinstance(decl.get_pointee().kind, cindex.TypeKind)
    ):
        ctype = decl.get_pointee().kind.spelling.lower()
        return dtype_to_ctype_registry().get_or_register_dtype(ctype)
    if isinstance(decl.kind, cindex.TypeKind):
        ctype = decl.kind.spelling.lower()
        return dtype_to_ctype_registry().get_or_register_dtype(ctype)
    if decl.kind == cindex.TypeKind.CONSTANTARRAY:
        return _get_dtype_from_decl_type(decl.get_array_element_type())
    raise NotImplementedError(f"_get_dtype_from_decl: {decl}")


class CToLoopyExpressionMapper(IdentityMapper):
    """Functions for mapping C expressions to Loopy"""

    def map_var_decl(self, expr: cindex.Cursor):
        "Maps variable declaration expression"
        (literal,) = expr.get_children()
        (val,) = literal.get_tokens()
        return (
            dtype_to_ctype_registry()
            .get_or_register_dtype(expr.type.kind.spelling.lower())
            .type
        )(val.spelling)

    def map_decl_ref_expr(self, expr: cindex.CursorKind):
        """Maps reference to a C declaration"""
        (var_name,) = expr.get_tokens()
        return prim.Variable(var_name.spelling)

    def map_array_subscript_expr(self, expr: cindex.CursorKind):
        """Maps C array reference"""
        if expr.kind == cindex.CursorKind.ARRAY_SUBSCRIPT_EXPR:
            var_name, index = expr.get_children()
            return prim.Subscript(self.rec(var_name), self.rec(index))
        if isinstance(expr.name, cindex.CursorKind):
            return prim.Subscript(
                self.rec(expr.name.name),
                (self.rec(expr.name.subscript), self.rec(expr.subscript)),
            )
        raise NotImplementedError

    def map_initlist(self, expr: cindex.CursorKind):
        """Maps C list initialization"""
        raise SyntaxError

    def map_unexposed_expr(self, expr: cindex.CursorKind):
        """Maps unexposed expression"""
        (child,) = expr.get_children()
        return self.rec(child)

    def map_binary_operator(self, expr: cindex.CursorKind):
        """Maps C binary operation"""
        ops = ""
        for i in expr.get_tokens():
            if i.spelling in _OPS:
                ops = i.spelling
                break
        oprtr = _C_BIN_OPS_TO_PYMBOLIC_OPS[ops]
        left, right = expr.get_children()
        return oprtr(self.rec(left), self.rec(right))


# @dataclass automatically adds special methods like __init__ and __repr__
@dataclass
class CToLoopyMapperContext:
    """Record expression context information"""

    # We don't need inner/outer inames. User should do these
    # transformations with loopy API. So we should just track inames.
    inames: FrozenSet[str]
    value_args: FrozenSet[str]
    predicates: FrozenSet[Union[prim.Comparison, prim.LogicalNot]]
    name_gen: UniqueNameGenerator

    def copy(self, **kwargs) -> "CToLoopyMapperContext":
        """Update class variables"""
        return replace(self, **kwargs)


@dataclass
class CToLoopyMapperAccumulator:
    """Record C expressions"""

    domains: List[isl.BasicSet]
    statements: List[lp.InstructionBase]
    kernel_data: List[Union[lp.ValueArg, lp.TemporaryVariable]]

    def copy(self, *, domains=None, statements=None, kernel_data=None):
        """Update class variables"""
        domains = domains or self.domains
        statements = statements or self.statements
        kernel_data = kernel_data or self.kernel_data
        return CToLoopyMapperAccumulator(domains, statements, kernel_data)


class CToLoopyLoopBoundMapper(CToLoopyExpressionMapper):
    """Map loop bounds"""

    def map_binaryop(self, expr: cindex.CursorKind):
        """Maps C binary operation"""
        oprtr = _C_BIN_OPS_TO_PYMBOLIC_OPS["//" if expr.op == "/" else expr.op]
        return oprtr(self.rec(expr.left), self.rec(expr.right))


# Helper function for parsing for loop kernels
def check_and_parse_for(
    expr: cindex.CursorKind, context: CToLoopyMapperContext
):
    """Parse for loop to retrieve loop variable, lower bound and upper bound"""
    decl_stmt, cond, ex, *_ = expr.get_children()
    (var_decl,) = decl_stmt.get_children()
    iname = var_decl.spelling

    # Sanity checks
    if not expr:
        raise NotImplementedError(
            "More than one initialization declarations not yet supported."
        )

    ops = []
    for i in cond.get_tokens():
        ops.append(i.spelling)
    exs = []
    for i in ex.get_tokens():
        exs.append(i.spelling)

    if not (
        cond.kind == cindex.CursorKind.BINARY_OPERATOR
        and "<" in ops
        and ops[0] == iname
    ):
        raise NotImplementedError("Only increasing domains are supported")

    _, right = cond.get_children()
    if ex.kind == cindex.CursorKind.UNARY_OPERATOR and "++" in exs:
        lbound_expr = CToLoopyLoopBoundMapper()(var_decl)
        ubound_expr = CToLoopyLoopBoundMapper()(right)
    else:
        raise NotImplementedError("Only increments by 1 are supported")

    if not (
        isinstance(expr.cond.left, c_ast.ID) and expr.cond.left.name == iname
    ) or not (
        isinstance(expr.next.expr, c_ast.ID) and expr.next.expr.name == iname
    ):
        raise SyntaxError(
            "Loop variable has to be the same in for condition and next"
            " operation"
        )

    return (iname, lbound_expr, ubound_expr)


class CToLoopyMapper(IdentityMapper):
    """Map C expressions"""

    def combine(self, values):
        """Combine mapped expressions"""
        # FIXME: Needs slightly more sophisticated checks here..
        # For example.
        # 1. No domain should have the dim_sets repeated.
        # 2. No arguments should have conflicting infos. like either being an
        #    ArrayArg or a ValueArg.

        new_domains = sum((value.domains for value in values), start=[])
        new_statements = sum((value.statements for value in values), start=[])
        new_kernel_data = sum((value.kernel_data for value in values), start=[])
        return CToLoopyMapperAccumulator(
            new_domains, new_statements, new_kernel_data
        )

    def map_if_stmt(
        self, expr: cindex.CursorKind, context: CToLoopyMapperContext
    ):
        """Map C if condition"""
        cond_expr, if_true, *rest = expr.get_children()
        cond = CToLoopyExpressionMapper()(cond_expr)
        # Map expr.iftrue with cond
        true_predicates = context.predicates | {cond}
        true_context = context.copy(predicates=true_predicates)
        true_result = self.rec(if_true, true_context)

        # Map expr.iffalse with !cond
        try:
            (if_false,) = rest
            false_predicates = context.predicates | {prim.LogicalNot(cond)}
            false_context = context.copy(predicates=false_predicates)
            false_result = self.rec(if_false, false_context)

            true_insn_ids, false_insn_ids = frozenset(), frozenset()
            for stmt in true_result.statements:
                true_insn_ids = true_insn_ids | {(stmt.id, "any")}
            for stmt in false_result.statements:
                false_insn_ids = false_insn_ids | {(stmt.id, "any")}

            for i, stmt in enumerate(true_result.statements):
                stmt.no_sync_with = false_insn_ids
                true_result.statements[i] = stmt
            for i, stmt in enumerate(false_result.statements):
                stmt.no_sync_with = true_insn_ids
                false_result.statements[i] = stmt
            return self.combine([true_result, false_result])
        except:
            return true_result

    def map_for_stmt(
        self, expr: cindex.CursorKind, context: CToLoopyMapperContext
    ):
        """Map C for loop"""
        _, _, _, body = expr.get_children()
        (iname, lbound_expr, ubound_expr) = check_and_parse_for(expr, context)

        space = isl.Space.create_from_names(
            isl.DEFAULT_CONTEXT, set=[iname], params=context.value_args
        )

        domain = make_slab(
            space,
            iname,
            aff_from_expr(space, lbound_expr),
            aff_from_expr(space, ubound_expr),
        )

        new_inames = context.inames | {iname}
        new_context = context.copy(inames=new_inames)

        children_result = self.rec(body, new_context)

        return children_result.copy(domains=children_result.domains + [domain])

    def map_decl_stmt(
        self, expr: cindex.CursorKind, context: CToLoopyMapperContext
    ):
        """Map C variable declaration"""
        (decl,) = expr.get_children()
        name = decl.spelling
        if decl.kind == cindex.CursorKind.VAR_DECL:
            shape = ()
        # In case expr is an array
        elif decl.kind == cindex.TypeKind.CONSTANTARRAY:
            # 1D array
            dims = [expr.type.dim]
            # 2D array
            if isinstance(expr.type.type, cindex.CursorKind):
                dims += [expr.type.type.dim]
            elif not isinstance(expr.type.type, cindex.CursorKind):
                raise NotImplementedError
            shape = tuple([CToLoopyExpressionMapper()(dim) for dim in dims])
        else:
            raise NotImplementedError

        if decl:
            lhs = prim.Variable(name)
            (right,) = decl.get_children()
            rhs = CToLoopyExpressionMapper()(right)
            return CToLoopyMapperAccumulator(
                [],
                [
                    lp.Assignment(
                        lhs,
                        expression=rhs,
                        within_inames=context.inames,
                        predicates=context.predicates,
                        id=context.name_gen(LOOPY_INSN_PREFIX),
                    )
                ],
                [
                    lp.TemporaryVariable(
                        name,
                        dtype=_get_dtype_from_decl_type(decl.type),
                        shape=shape,
                    )
                ],
            )
        return CToLoopyMapperAccumulator(
            [],
            [],
            [
                lp.TemporaryVariable(
                    name,
                    dtype=_get_dtype_from_decl_type(expr.type),
                    shape=shape,
                )
            ],
        )

    def map_compound_stmt(
        self,
        expr: cindex.CursorKind.COMPOUND_STMT,
        context: CToLoopyMapperContext,
    ):
        """Map C compound expression"""
        if expr.get_children() is not None:
            return self.combine(
                [self.rec(child, context) for child in expr.get_children()]
            )
        return CToLoopyMapperAccumulator([], [], [])

    def map_binary_operator(
        self, expr: cindex.CursorKind, context: CToLoopyMapperContext
    ):
        """Maps C binary operation"""
        left, right = expr.get_children()

        lhs = CToLoopyExpressionMapper()(left)
        rhs = CToLoopyExpressionMapper()(right)

        # FIXME: This could be sharpened so that other instruction types are
        # also emitted.
        return CToLoopyMapperAccumulator(
            [],
            [
                lp.Assignment(
                    lhs,
                    expression=rhs,
                    within_inames=context.inames,
                    predicates=context.predicates,
                    id=context.name_gen(LOOPY_INSN_PREFIX),
                )
            ],
            [],
        )


@dataclass
class ExternalContext:
    "Maintain information on variable declaration in a function"
    function_name: Optional[str]
    var_to_decl: Dict[str, cindex.Cursor]

    def copy(self, **kwargs):
        """Get external context with updated variable values"""
        updated_kwargs = kwargs.copy()
        for field in fields(self):
            if field.name not in updated_kwargs:
                updated_kwargs[field.name] = getattr(self, field.name)

        return ExternalContext(**updated_kwargs)

    def add_decl(self, var_name: str, decl: cindex.Cursor) -> None:
        """Adds kernel parameter variable declaration to the context"""
        new_var_to_decl = self.var_to_decl.copy()
        new_var_to_decl[var_name] = decl
        return self.copy(var_to_decl=new_var_to_decl)

    def var_to_dtype(self, var_name: str) -> None:
        """Converts variable to data type"""
        return _get_dtype_from_decl_type(self.var_to_decl[var_name].type)

    def get_typedecl(self):
        """Get type declarations"""
        return {
            k: v
            for k, v in self.var_to_decl.items()
            if v.type.kind != cindex.TypeKind.POINTER
        }


def decl_to_knl_arg(decl: cindex.Cursor, dtype):
    """Type declaration to kernel arg"""
    decl_type = decl.type.kind
    if decl_type == cindex.TypeKind.POINTER:
        return lp.ArrayArg(
            decl.spelling, dtype=dtype, address_space=AddressSpace.GLOBAL
        )
    if isinstance(decl_type, cindex.TypeKind):
        return lp.ValueArg(decl.spelling, dtype=dtype)
    raise NotImplementedError(f"decl_to_knl_arg: {decl} is invalid.")


def c_to_loopy(c_str: str, backend: str) -> lp.translation_unit.TranslationUnit:
    """Map C kernel to Loopy"""
    index = cindex.Index.create()
    translation_unit = index.parse("foo.c", unsaved_files=[("foo.c", c_str)])
    (node,) = translation_unit.cursor.get_children()

    # Init `var_to_decl` based on function parameters
    context = ExternalContext(function_name=node.spelling, var_to_decl={})
    knl_args = []
    body = None
    for arg in node.get_children():
        if arg.kind == cindex.CursorKind.PARM_DECL:
            context = context.add_decl(arg.spelling, arg)
            knl_args.append(
                decl_to_knl_arg(arg, context.var_to_dtype(arg.spelling))
            )
        body = arg

    # Map C for loop to loopy kernel
    acc = CToLoopyMapper()(
        body,
        CToLoopyMapperContext(
            frozenset(),
            context.get_typedecl().keys(),
            frozenset(),
            UniqueNameGenerator(),
        ),
    )

    unique_domains = frozenset()
    for domain in acc.domains:
        new = True
        for unique_domain in unique_domains:
            if unique_domain == domain:
                new = False
                for i, j in zip(
                    unique_domain.get_id_dict().keys(),
                    domain.get_id_dict().keys(),
                ):
                    if i.get_name() != j.get_name():
                        new = True
                        break
        if new:
            unique_domains = unique_domains | {domain}

    knl = lp.make_kernel(
        unique_domains,
        acc.statements,
        knl_args + acc.kernel_data + [...],
        lang_version=LOOPY_LANG_VERSION,
        name=node.spelling,
        seq_dependencies=True,
        target=_BACKEND_TO_TARGET[backend],
    )

    return knl


if __name__ == "__main__":
    import os
    import sys

    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    KNL_STR = """
          void foo(double *a, int N) {
            for (int i = 0; i < N; i++)
              a[i] = i + 7;
          }
          """
    lp_knl = c_to_loopy(KNL_STR, "cuda")
    print(lp.generate_code_v2(lp_knl).device_code())
