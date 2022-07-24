from dataclasses import dataclass, fields
from typing import Dict, FrozenSet, List, Optional, Union

import islpy as isl
import loopy as lp
import pycparser.c_ast as c_ast
import pymbolic.primitives as prim
from loopy.isl_helpers import make_slab
from loopy.symbolic import aff_from_expr
from pycparser import c_parser
from pytools import UniqueNameGenerator, memoize

LOOPY_LANG_VERSION = (2018, 2)
LOOPY_INSN_PREFIX = "_nomp_insn"

_C_BIN_OPS_TO_PYMBOLIC_OPS = {
    "*": lambda x, y: prim.Product((x, y)),
    "+": lambda x, y: prim.Sum((x, y)),
    "-": lambda l, r: prim.Sum((l, prim.Product((-1, r)))),
    "/": lambda l, r: prim.Quotient(l, r),
    "//": lambda l, r: prim.FloorDiv(l, r),
    "%": lambda l, r: prim.Remainder(l, r),
    "<": lambda l, r: prim.Comparison(l, "<", r),
    "<=": lambda l, r: prim.Comparison(l, "<=", r),
    ">": lambda l, r: prim.Comparison(l, ">", r),
    ">=": lambda l, r: prim.Comparison(l, ">=", r),
    "==": lambda l, r: prim.Comparison(l, "==", r),
    "!=": lambda l, r: prim.Comparison(l, "!=", r),
}


class IdentityMapper:
    def rec(self, node, *args, **kwargs):
        """Visit a node."""
        try:
            mapper_method = getattr(self, "map_" + node.__class__.__name__)
        except AttributeError:
            mapper_method = getattr(self, "map_generic_ast_node")

        return mapper_method(node, *args, **kwargs)

    __call__ = rec

    def map_generic_ast_node(
        self, node: c_ast.Node, *args, **kwargs
    ) -> c_ast.Node:
        new_kwargs = {}
        for attr_name in node.__slots__:
            if attr_name != "__weakref__":
                attr = getattr(node, attr_name)
                new_kwargs[attr_name] = self.rec(attr, *args, **kwargs)

        return type(node)(**new_kwargs)

    def map_NoneType(self, node, *args, **kwargs) -> None:
        return None

    def map_str(self, node: str, *args, **kwargs) -> str:
        return node

    def map_int(self, node: int, *args, **kwargs) -> int:
        return node

    def map_list(self, node: list, *args, **kwargs) -> list:
        return [self.rec(c, *args, **kwargs) for c in node]


# Memoize is caching the return of a function based on its input parameters
# so we use it here instead of general @cache.
@memoize
def dtype_to_ctype_registry():
    from compyte.dtypes import DTypeRegistry, fill_registry_with_c_types

    dtype_reg = DTypeRegistry()
    fill_registry_with_c_types(dtype_reg, True)
    return dtype_reg


@memoize
def _get_dtype_from_decl_type(decl_type):
    if (
        isinstance(decl_type, c_ast.PtrDecl)
        and isinstance(decl_type.type, c_ast.TypeDecl)
        and isinstance(decl_type.type.type, c_ast.IdentifierType)
        and isinstance(decl_type.type.type.names, list)
        and len(decl_type.type.type.names) == 1
    ):
        (ctype,) = decl_type.type.type.names
        return dtype_to_ctype_registry().get_or_register_dtype(ctype)
    elif (
        isinstance(decl_type, c_ast.TypeDecl)
        and isinstance(decl_type.type, c_ast.IdentifierType)
        and isinstance(decl_type.type.names, list)
        and len(decl_type.type.names) == 1
    ):
        (ctype,) = decl_type.type.names
        return dtype_to_ctype_registry().get_or_register_dtype(ctype)
    elif isinstance(decl_type, c_ast.ArrayDecl):
        return _get_dtype_from_decl_type(decl_type.type)

    raise NotImplementedError(decl_type)


class CToLoopyExpressionMapper(IdentityMapper):
    def map_Constant(self, expr: c_ast.Constant):
        return (
            dtype_to_ctype_registry().get_or_register_dtype(expr.type).type
        )(expr.value)

    def map_BinaryOp(self, expr: c_ast.BinaryOp):
        op = _C_BIN_OPS_TO_PYMBOLIC_OPS[expr.op]
        return op(self.rec(expr.left), self.rec(expr.right))

    def map_ID(self, expr: c_ast.ID):
        return prim.Variable(expr.name)

    def map_ArrayRef(self, expr: c_ast.ArrayRef):
        if isinstance(expr.name, c_ast.ID):
            return prim.Subscript(self.rec(expr.name), self.rec(expr.subscript))
        elif isinstance(expr.name, c_ast.ArrayRef):
            return prim.Subscript(
                self.rec(expr.name.name),
                (self.rec(expr.name.subscript), self.rec(expr.subscript)),
            )
        else:
            raise NotImplementedError

    def map_InitList(self, expr: c_ast.InitList):
        raise SyntaxError

    def map_generic_ast_node(self, expr: c_ast.Node):
        raise NotImplementedError


# @dataclass automatically adds special methods like __init__ and __repr__
@dataclass
class CToLoopyMapperContext:
    # We don't need inner/outer inames. User should do these
    # transformations with loopy API. So we should just track inames.
    inames: FrozenSet[str]
    value_args: FrozenSet[str]
    predicates: FrozenSet[Union[prim.Comparison, prim.LogicalNot]]
    name_gen: UniqueNameGenerator

    def copy(self, **kwargs) -> "CToLoopyMapperContext":
        from dataclasses import replace

        return replace(self, **kwargs)


@dataclass
class CToLoopyMapperAccumulator:
    domains: List[isl.BasicSet]
    statements: List[lp.InstructionBase]
    kernel_data: List[Union[lp.ValueArg, lp.TemporaryVariable]]
    arguments: FrozenSet[lp.KernelArgument]

    def copy(
        self,
        *,
        domains=None,
        statements=None,
        kernel_data=None,
        arguments=None,
    ):
        domains = domains or self.domains
        statements = statements or self.statements
        kernel_data = kernel_data or self.kernel_data
        arguments = arguments or self.arguments

        return CToLoopyMapperAccumulator(
            domains, statements, kernel_data, arguments
        )


class CToLoopyLoopBoundMapper(CToLoopyExpressionMapper):
    def map_BinaryOp(self, expr: c_ast.BinaryOp):
        op = _C_BIN_OPS_TO_PYMBOLIC_OPS["//" if expr.op == "/" else expr.op]
        return op(self.rec(expr.left), self.rec(expr.right))


# Helper function for parsing for loop kernels
def check_and_parse_for(expr: c_ast.For, context: CToLoopyMapperContext):
    (init_decl,) = expr.init.decls
    iname = init_decl.name

    # Sanity checks
    if len(expr.init.decls) != 1:
        raise NotImplementedError(
            "More than one initialization" " declarations not yet supported."
        )

    if not (
        isinstance(expr.cond, c_ast.BinaryOp)
        and expr.cond.op == "<"
        and isinstance(expr.cond.left, c_ast.ID)
        and expr.cond.left.name == iname
    ):
        raise NotImplementedError("Only increasing domains are supported")

    if isinstance(expr.next, c_ast.UnaryOp) and expr.next.op in {"p++", "++p"}:
        lbound_expr = CToLoopyLoopBoundMapper()(init_decl.init)
        ubound_expr = CToLoopyLoopBoundMapper()(expr.cond.right)
    else:
        raise NotImplementedError("Only increments by 1 are supported")

    return (iname, lbound_expr, ubound_expr)


class CToLoopyMapper(IdentityMapper):
    def combine(self, values):
        # FIXME: Needs slightly more sophisticated checks here..
        # For example.
        # 1. No domain should have the dim_sets repeated.
        # 2. No arguments should have conflicting infos. like either being an
        #    ArrayArg or a ValueArg.

        from functools import reduce

        new_domains = sum((value.domains for value in values), start=[])
        new_statements = sum((value.statements for value in values), start=[])
        new_kernel_data = sum((value.kernel_data for value in values), start=[])
        new_args = reduce(
            frozenset.union, (value.arguments for value in values), frozenset()
        )
        return CToLoopyMapperAccumulator(
            new_domains, new_statements, new_kernel_data, new_args
        )

    def map_If(self, expr: c_ast.If, context: CToLoopyMapperContext):
        cond = CToLoopyExpressionMapper()(expr.cond)

        # Map expr.iftrue with cond
        true_predicates = context.predicates | {cond}
        true_context = context.copy(predicates=true_predicates)
        true_result = self.rec(expr.iftrue, true_context)

        # Map expr.iffalse with !cond
        if expr.iffalse:
            false_predicates = context.predicates | {prim.LogicalNot(cond)}
            false_context = context.copy(predicates=false_predicates)
            false_result = self.rec(expr.iffalse, false_context)

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
        return true_result

    def map_For(self, expr: c_ast.For, context: CToLoopyMapperContext):
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
        children_result = self.rec(expr.stmt, new_context)

        return children_result.copy(domains=children_result.domains + [domain])

    def map_Decl(self, expr: c_ast.Decl, context: CToLoopyMapperContext):
        name = expr.name
        if isinstance(expr.type, c_ast.TypeDecl):
            shape = ()
        # In case expr is an array
        elif isinstance(expr.type, c_ast.ArrayDecl):
            # 1D array
            dims = [expr.type.dim]
            # 2D array
            if isinstance(expr.type.type, c_ast.ArrayDecl):
                dims += [expr.type.type.dim]
            elif not isinstance(expr.type.type, c_ast.TypeDecl):
                raise NotImplementedError
            shape = tuple([CToLoopyExpressionMapper()(dim) for dim in dims])
        else:
            raise NotImplementedError

        if expr.init is not None:
            lhs = CToLoopyExpressionMapper()(c_ast.ID(name))
            rhs = CToLoopyExpressionMapper()(expr.init)
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
                        dtype=_get_dtype_from_decl_type(expr.type),
                        shape=shape,
                    )
                ],
                frozenset(),
            )
        else:
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
                frozenset(),
            )

    def map_Compound(
        self, expr: c_ast.Compound, context: CToLoopyMapperContext
    ):
        if expr.block_items is not None:
            return self.combine(
                [self.rec(child, context) for child in expr.block_items]
            )
        else:
            return CToLoopyMapperAccumulator([], [], [], frozenset())

    def map_Assignment(
        self, expr: c_ast.Assignment, context: CToLoopyMapperContext
    ):
        lhs = CToLoopyExpressionMapper()(expr.lvalue)
        rhs = CToLoopyExpressionMapper()(expr.rvalue)

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
            frozenset(),
        )

    def map_InitList(
        self, expr: c_ast.InitList, context: CToLoopyMapperContext
    ):
        raise SyntaxError


@dataclass
class ExternalContext:
    function_name: Optional[str]
    var_to_decl: Dict[str, c_ast.Decl]

    def copy(self, **kwargs):
        updated_kwargs = kwargs.copy()
        for field in fields(self):
            if field.name not in updated_kwargs:
                updated_kwargs[field.name] = getattr(self, field.name)

        return ExternalContext(**updated_kwargs)

    def add_decl(self, var_name: str, decl: c_ast.Decl) -> None:
        new_var_to_decl = self.var_to_decl.copy()
        new_var_to_decl[var_name] = decl
        return self.copy(var_to_decl=new_var_to_decl)

    def var_to_dtype(self, var_name: str) -> None:
        return _get_dtype_from_decl_type(self.var_to_decl[var_name].type)

    def get_typedecl(self):
        return {
            k: v
            for k, v in self.var_to_decl.items()
            if isinstance(v.type, c_ast.TypeDecl)
        }


def c_to_loopy(c_str: str):
    # Parse the function
    parser = c_parser.CParser()
    ast = parser.parse(c_str)
    node = ast.ext[0]

    # Init `var_to_decl` based on function parameters
    context = ExternalContext(function_name=node.decl.name, var_to_decl={})
    for arg in node.decl.type.args.params:
        if isinstance(arg, c_ast.Decl):
            context = context.add_decl(arg.name, arg)

    # Map C for loop to loopy kernel
    acc = CToLoopyMapper()(
        node.body,
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
        for d in unique_domains:
            if d == domain:
                new = False
                for (i, j) in zip(
                    d.get_id_dict().keys(), domain.get_id_dict().keys()
                ):
                    if i.get_name() != j.get_name():
                        new = True
                        break
        if new:
            unique_domains = unique_domains | {domain}

    knl = lp.make_kernel(
        unique_domains,
        acc.statements,
        acc.kernel_data + [...],
        lang_version=LOOPY_LANG_VERSION,
        name=node.decl.name,
        seq_dependencies=True,
    )

    knl = lp.add_dtypes(
        knl,
        {
            arg.name: context.var_to_dtype(arg.name)
            for arg in knl.default_entrypoint.args
        },
    )
    return knl


if __name__ == "__main__":
    import os
    import sys

    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    knl = """
          void foo(double *a, int N) {
            for (int i = 0; i < N; i++)
              a[i] = i;
          }
          """
    foo(knl)
