"""Loopy API wrapper"""
import hashlib
from dataclasses import dataclass, replace
from typing import Dict, FrozenSet, List, Union

import islpy as isl
import loopy as lp
import numpy as np
import pymbolic.primitives as prim
from clang import cindex
from kernel_wrappers import (
    BaseKernelWrapper,
    ISPCKernelWrapper,
    SyclKernelWrapper,
)
from loopy.isl_helpers import make_slab
from loopy.kernel.data import AddressSpace
from loopy.symbolic import aff_from_expr
from loopy.target.c.compyte.dtypes import (
    DTypeRegistry,
    fill_registry_with_c_types,
)
from pytools import UniqueNameGenerator

LOOPY_LANG_VERSION = (2018, 2)
LOOPY_INSN_PREFIX = "_nomp_insn"

_C_OPS_TO_PYMBOLIC_OPS = {
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
    "&&": lambda l, r: prim.LogicalAnd((l, r)),
    "||": lambda l, r: prim.LogicalOr((l, r)),
    "&": lambda l, r: prim.BitwiseAnd((l, r)),
    "|": lambda l, r: prim.BitwiseOr((l, r)),
    "^": lambda l, r: prim.BitwiseXor((l, r)),
    "<<": prim.LeftShift,
    ">>": prim.RightShift,
}
_C_UNARY_OPS_TO_PYMBOLIC_OPS = {
    "~": prim.BitwiseNot,
    "!": prim.LogicalNot,
}
_CLANG_TYPE_TO_C_TYPE = {
    "bool": "bool",
    "char_s": "signed char",
    "char_u": "unsigned char",
    "short": "short",
    "ushort": "unsigned short",
    "int": "int",
    "uint": "unsigned int",
    "long": "long",
    "ulong": "unsigned long",
    "float": "float",
    "double": "double",
    "uchar": "unsigned char",
    "schar": "signed char",
}
_BACKEND_TO_TARGET = {
    "opencl": lp.OpenCLTarget(),
    "cuda": lp.CudaTarget(),
    "ispc": lp.ISPCTarget(),
    "hip": lp.CudaTarget(),
    "sycl": lp.SYCLTarget(),
}
_BACKEND_TO_WRAPPER = {
    "ispc": ISPCKernelWrapper(prefix="nomp_ispc"),
    "sycl": SyclKernelWrapper(prefix="nomp_sycl", includes="CL/sycl.hpp"),
}
_ARRAY_TYPES = [cindex.TypeKind.CONSTANTARRAY, cindex.TypeKind.INCOMPLETEARRAY]
_ARRAY_TYPES_W_PTR = _ARRAY_TYPES + [cindex.TypeKind.POINTER]


# pylint-disable-reason: Prefer the Singleton pattern instead of a global
# variable.
# pylint: disable=R0903
class DtypeRegAcc:
    """Accessor class for DTypeRegistry"""

    __instance = None

    def __init__(self):
        if DtypeRegAcc.__instance is None:
            DtypeRegAcc.__instance = self
            self.dtype_reg = DTypeRegistry()
            fill_registry_with_c_types(self.dtype_reg, True)

    @staticmethod
    def get_or_register_dtype(ctype: str) -> np.dtype:
        """Return a dtype given a ctype."""
        if DtypeRegAcc.__instance is None:
            DtypeRegAcc()
        return DtypeRegAcc.__instance.dtype_reg.get_or_register_dtype(
            _CLANG_TYPE_TO_C_TYPE[ctype]
        )


class IdentityMapper:
    """AST node mapper."""

    def rec(self, node, *args, **kwargs):
        """Visit a node in the AST."""
        class_name = str(node.kind).split(".")[1]
        try:
            mapper_method = getattr(self, "map_" + class_name.lower())
        except AttributeError as exc:
            raise NotImplementedError(
                f"Mapper method for {class_name} is not implemented."
            ) from exc
        return mapper_method(node, *args, **kwargs)

    __call__ = rec

    def map_integer_literal(
        self, expr: cindex.CursorKind.INTEGER_LITERAL
    ) -> np.dtype:
        """Maps a integer literal."""
        (val,) = expr.get_tokens()
        ctype = expr.type.kind.spelling.lower()
        return (DtypeRegAcc.get_or_register_dtype(ctype).type)(val.spelling)

    def map_floating_literal(
        self, expr: cindex.CursorKind.FLOATING_LITERAL
    ) -> np.dtype:
        """Maps a float literal."""
        (val,) = expr.get_tokens()
        ctype = expr.type.kind.spelling.lower()
        return (DtypeRegAcc.get_or_register_dtype(ctype).type)(val.spelling)


def _get_dtype_from_decl_type(decl) -> np.dtype:
    """Retrieve data type from declaration."""
    if (
        isinstance(decl.kind, cindex.TypeKind)
        and decl.kind == cindex.TypeKind.POINTER
        and isinstance(decl.get_pointee().kind, cindex.TypeKind)
    ):
        ctype = decl.get_pointee().kind.spelling.lower()
        return DtypeRegAcc.get_or_register_dtype(ctype)
    if decl.kind in _ARRAY_TYPES:
        return _get_dtype_from_decl_type(decl.get_array_element_type())
    if isinstance(decl.kind, cindex.TypeKind):
        ctype = decl.kind.spelling.lower()
        return DtypeRegAcc.get_or_register_dtype(ctype)
    raise NotImplementedError(f"_get_dtype_from_decl: {decl}")


def get_op_str(expr: cindex.CursorKind, lhs: cindex.CursorKind) -> str:
    """Get operator as string from a binary expression."""
    return list(expr.get_tokens())[len(list(lhs.get_tokens()))].spelling


class CToLoopyExpressionMapper(IdentityMapper):
    """Functions for mapping C expressions to Loopy expressions."""

    def map_decl_ref_expr(
        self, expr: cindex.CursorKind.DECL_REF_EXPR
    ) -> prim.Variable:
        """Maps a variable reference to a C declaration."""
        (var_name,) = expr.get_tokens()
        return prim.Variable(var_name.spelling)

    def map_array_subscript_expr(
        self, expr: cindex.CursorKind.ARRAY_SUBSCRIPT_EXPR
    ) -> prim.Subscript:
        """Maps a C array subscript expression."""
        (unexpsd_expr, index) = expr.get_children()
        (child,) = unexpsd_expr.get_children()
        if child.kind == cindex.CursorKind.DECL_REF_EXPR:
            return prim.Subscript(self.rec(child), self.rec(index))
        if child.kind == cindex.CursorKind.ARRAY_SUBSCRIPT_EXPR:
            (unexpsd_expr_2, index_2) = child.get_children()
            (child_2,) = unexpsd_expr_2.get_children()
            return prim.Subscript(
                self.rec(child_2), (self.rec(index_2), self.rec(index))
            )
        raise NotImplementedError(
            f"{child.kind} is not recognized as a child of a"
            " ARRAY_SUBSCRIPT_EXPR"
        )

    def map_unary_operator(
        self, expr: cindex.CursorKind.UNARY_OPERATOR
    ) -> prim.Expression:
        """Maps a C unary operator."""
        exs = [token.spelling for token in expr.get_tokens()]
        op_str = exs[0] if exs[0] in _C_UNARY_OPS_TO_PYMBOLIC_OPS else exs[-1]
        try:
            oprtr = _C_UNARY_OPS_TO_PYMBOLIC_OPS[op_str]
        except KeyError as exc:
            raise SyntaxError(f"Invalid unary operator: {op_str}.") from exc
        (var,) = expr.get_children()
        return oprtr(self.rec(var))

    def map_binary_operator(
        self, expr: cindex.CursorKind.BINARY_OPERATOR
    ) -> prim.Expression:
        """Maps a C binary operator."""
        left, right = expr.get_children()
        op_str = get_op_str(expr, left)
        try:
            oprtr = _C_OPS_TO_PYMBOLIC_OPS[op_str]
        except KeyError as exc:
            raise SyntaxError(f"Invalid binary operator: {op_str}.") from exc
        return oprtr(self.rec(left), self.rec(right))

    def map_unexposed_expr(
        self, expr: cindex.CursorKind.UNEXPOSED_EXPR
    ) -> prim.Expression:
        """Maps an unexposed expression."""
        (child,) = expr.get_children()
        return self.rec(child)

    def map_paren_expr(
        self, expr: cindex.CursorKind.PAREN_EXPR
    ) -> prim.Expression:
        """Maps a parenthesized expression."""
        (child,) = expr.get_children()
        return self.rec(child)

    def map_conditional_operator(
        self, expr: cindex.CursorKind.CONDITIONAL_OPERATOR
    ) -> tuple[prim.Expression]:
        """Maps an ternary operator expression."""
        cond_expr, if_true_right, if_false_right = expr.get_children()
        cond = CToLoopyExpressionMapper()(cond_expr)
        if_true_rhs = CToLoopyExpressionMapper()(if_true_right)
        if_false_rhs = CToLoopyExpressionMapper()(if_false_right)
        return cond, if_true_rhs, if_false_rhs


@dataclass
class CToLoopyMapperContext:
    """Keep track of kernel context information."""

    # We don't need inner/outer inames. User should do these
    # transformations with loopy API. So we should just track inames.
    inames: FrozenSet[str]
    value_args: FrozenSet[str]
    predicates: FrozenSet[Union[prim.Comparison, prim.LogicalNot]]
    name_gen: UniqueNameGenerator

    def copy(self, **kwargs) -> "CToLoopyMapperContext":
        """Return a copy of the context by replacing fields."""
        return replace(self, **kwargs)


@dataclass
class CToLoopyMapperAccumulator:
    """Record C statements and expressions."""

    domains: List[isl.BasicSet]
    statements: List[lp.InstructionBase]
    kernel_data: List[Union[lp.ValueArg, lp.TemporaryVariable]]

    def copy(self, *, domains=None, statements=None, kernel_data=None):
        """Return a copy of the current object."""
        domains = domains or self.domains
        statements = statements or self.statements
        kernel_data = kernel_data or self.kernel_data
        return CToLoopyMapperAccumulator(domains, statements, kernel_data)


def set_result_stmts(results: tuple[CToLoopyMapperAccumulator]) -> tuple:
    """Set insn ids for results."""
    true_insn_ids, false_insn_ids = frozenset(), frozenset()
    for stmt in results[0].statements:
        true_insn_ids = true_insn_ids | {(stmt.id, "any")}
    for stmt in results[1].statements:
        false_insn_ids = false_insn_ids | {(stmt.id, "any")}

    for i, stmt in enumerate(results[0].statements):
        stmt.no_sync_with = false_insn_ids
        results[0].statements[i] = stmt
    for i, stmt in enumerate(results[1].statements):
        stmt.no_sync_with = true_insn_ids
        results[1].statements[i] = stmt
    return results[0], results[1]


class CToLoopyLoopBoundMapper(CToLoopyExpressionMapper):
    """Map loop bounds."""

    def map_binary_operator(self, expr: cindex.CursorKind) -> prim.Expression:
        """Maps a C binary operator."""
        (left, right) = expr.get_children()
        op_str = get_op_str(expr, left)
        try:
            oprtr = _C_OPS_TO_PYMBOLIC_OPS["//" if op_str == "/" else op_str]
        except KeyError as exc:
            raise SyntaxError(f"Invalid binary operator: {op_str}.") from exc
        return oprtr(self.rec(left), self.rec(right))


# pylint-disable-reason: check_and_parse can be split to two methods but it is
# not necessary to do so.
# pylint: disable=R0903
class ForLoopInfo:
    """Store meta information about For loops."""

    def __init__(self, expr: cindex.CursorKind.FOR_STMT):
        (self.decl, self.cond, self.update, self.body) = expr.get_children()

    def check_and_parse(self):
        """Check and parse For-loop components."""
        decls = list(self.decl.get_children())
        if len(decls) == 0:
            raise NotImplementedError(
                "For loop variable must be initialized in loop initializer."
            )
        if len(decls) > 1:
            raise NotImplementedError(
                f"Multiple variable initializations are not supported: {decls}"
            )

        cond_ok, cname, cleft, cright = False, None, None, None
        if self.cond.kind == cindex.CursorKind.BINARY_OPERATOR:
            (cleft, cright) = self.cond.get_children()
            if get_op_str(self.cond, cleft) in ["<", "<="]:
                cond_ok, cname = True, list(cleft.get_children())[0].spelling
        if not cond_ok:
            raise NotImplementedError("For loop condition must be < or <=.")

        update_ok, uname = False, None
        if self.update.kind == cindex.CursorKind.UNARY_OPERATOR:
            if "++" in [token.spelling for token in self.update.get_tokens()]:
                update_ok, uname = (
                    True,
                    list(self.update.get_children())[0].spelling,
                )
        if not update_ok:
            raise NotImplementedError("For loop update must be ++")

        if uname != cname:
            raise SyntaxError(
                "For loop variable must be same in initialization, condition"
                f" and increment. {uname} {cname}"
            )

        (var,) = decls[0].get_children()
        lbound = CToLoopyLoopBoundMapper()(var)
        ubound = CToLoopyLoopBoundMapper()(cright)
        return (cname, lbound, ubound, self.body)


def set_tf_results(
    lhs, rhs, context: CToLoopyMapperContext
) -> tuple[CToLoopyMapperAccumulator]:
    """Helper function to get true false results"""
    true_result = CToLoopyMapperAccumulator(
        [],
        [
            lp.Assignment(
                lhs,
                expression=rhs[1],
                within_inames=context.inames,
                predicates=context.copy(
                    predicates=context.predicates | {rhs[0]}
                ).predicates,
                id=context.name_gen(LOOPY_INSN_PREFIX),
            )
        ],
        [],
    )
    false_result = CToLoopyMapperAccumulator(
        [],
        [
            lp.Assignment(
                lhs,
                expression=rhs[2],
                within_inames=context.inames,
                predicates=context.copy(
                    predicates=context.predicates | {prim.LogicalNot(rhs[0])}
                ).predicates,
                id=context.name_gen(LOOPY_INSN_PREFIX),
            )
        ],
        [],
    )
    return true_result, false_result


class CToLoopyMapper(IdentityMapper):
    """Map C expressions and statemements to Loopy expressions."""

    def combine(self, values):
        """Combine mapped expressions."""
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
        self, expr: cindex.CursorKind.IF_STMT, context: CToLoopyMapperContext
    ) -> CToLoopyMapperAccumulator:
        """Maps a C if condition."""
        (cond_expr, if_true, *rest) = expr.get_children()

        # Map expr.iftrue with cond.
        cond = CToLoopyExpressionMapper()(cond_expr)
        true_result = self.rec(
            if_true, context.copy(predicates=context.predicates | {cond})
        )

        # Map expr.iffalse with !cond if it exists.
        if len(rest) > 0:
            (if_false,) = rest
            false_result = self.rec(
                if_false,
                context.copy(
                    predicates=context.predicates | {prim.LogicalNot(cond)}
                ),
            )

            return self.combine(set_result_stmts((true_result, false_result)))
        return true_result

    def map_for_stmt(
        self, expr: cindex.CursorKind.FOR_STMT, context: CToLoopyMapperContext
    ) -> CToLoopyMapperAccumulator:
        """Maps a C For loop."""
        (iname, lbound, ubound, body) = ForLoopInfo(expr).check_and_parse()

        space = isl.Space.create_from_names(
            isl.DEFAULT_CONTEXT, set=[iname], params=context.value_args
        )

        domain = make_slab(
            space,
            iname,
            aff_from_expr(space, lbound),
            aff_from_expr(space, ubound),
        )

        new_inames = context.inames | {iname}
        new_context = context.copy(inames=new_inames)

        children_result = self.rec(body, new_context)
        return children_result.copy(domains=children_result.domains + [domain])

    def map_decl_stmt(
        self, expr: cindex.CursorKind.DECL_STMT, context: CToLoopyMapperContext
    ) -> CToLoopyMapperAccumulator:
        """Maps a C declaration statements."""
        (decl,) = expr.get_children()
        return self.rec(decl, context)

    def map_var_decl(
        self, expr: cindex.CursorKind.VAR_DECL, context: CToLoopyMapperContext
    ) -> CToLoopyMapperAccumulator:
        """Maps a C variable declaration."""

        def check_and_parse_decl(expr: cindex.CursorKind):
            name, init = expr.spelling, None
            children = list(expr.get_children())
            if expr.type.kind == cindex.TypeKind.CONSTANTARRAY:
                dims = []
                for child in children:
                    if child.kind == cindex.CursorKind.INIT_LIST_EXPR:
                        init = child
                    # FIXME: This is wrong.
                    elif child.kind == cindex.CursorKind.INTEGER_LITERAL:
                        dims.append(child)
                    else:
                        raise NotImplementedError(
                            f"Unable to parse: {child.kind}"
                        )
                shape = tuple(CToLoopyExpressionMapper()(dim) for dim in dims)
                return (name, shape, init)
            if isinstance(expr.type.kind, cindex.TypeKind):
                if len(children) == 1:
                    init = children[0]
                return (name, (), init)
            raise NotImplementedError(f"Unable to parse: {expr.type.kind}")

        (name, shape, init) = check_and_parse_decl(expr)
        if init is not None:
            lhs = prim.Variable(name)
            rhs = CToLoopyExpressionMapper()(init)
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
    ) -> CToLoopyMapperAccumulator:
        """Maps a C compound expression."""
        if expr.get_children() is not None:
            return self.combine(
                [self.rec(child, context) for child in expr.get_children()]
            )
        return CToLoopyMapperAccumulator([], [], [])

    def map_binary_operator(
        self,
        expr: cindex.CursorKind.BINARY_OPERATOR,
        context: CToLoopyMapperContext,
    ) -> CToLoopyMapperAccumulator:
        """Maps a C binary operator."""
        left, right = expr.get_children()
        lhs = CToLoopyExpressionMapper()(left)
        rhs = CToLoopyExpressionMapper()(right)

        # Map assignment to an if-else if rhs is a conditional operator
        # statement
        if isinstance(rhs, tuple):
            return self.combine(
                set_result_stmts(set_tf_results(lhs, rhs, context))
            )
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

    def map_compound_assignment_operator(
        self,
        expr: cindex.CursorKind.COMPOUND_ASSIGNMENT_OPERATOR,
        context: CToLoopyMapperContext,
    ) -> CToLoopyMapperAccumulator:
        """Maps a C compound assignment operator."""
        left, right = expr.get_children()
        lhs = CToLoopyExpressionMapper()(left)
        rhs = CToLoopyExpressionMapper()(right)

        if isinstance(rhs, tuple):
            rhs = list(rhs)
            op_str = get_op_str(expr, left)
            for i in range(1, 3):
                if op_str == "+=":
                    rhs[i] = lhs + rhs[i]
                elif op_str == "-=":
                    rhs[i] = lhs - rhs[i]
                elif op_str == "*=":
                    rhs[i] = lhs * rhs[i]
                elif op_str != "=":
                    raise NotImplementedError(
                        f"Mapping not implemented for {op_str}"
                    )

            return self.combine(
                set_result_stmts(set_tf_results(lhs, rhs, context))
            )

        op_str = get_op_str(expr, left)
        if op_str == "+=":
            rhs = lhs + rhs
        elif op_str == "-=":
            rhs = lhs - rhs
        elif op_str == "*=":
            rhs = lhs * rhs
        elif op_str != "=":
            raise NotImplementedError(f"Mapping not implemented for {op_str}")

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

    # pylint-disable-reason: `expr` has to be present even though unused
    # since all the members of the mapper must have the same API
    def map_break_stmt(
        self,
        expr: cindex.CursorKind.BREAK_STMT,  # pylint: disable=W0613
        context: CToLoopyMapperContext,
    ) -> CToLoopyMapperAccumulator:
        """Maps a C break statement."""
        return CToLoopyMapperAccumulator(
            [],
            [
                lp.CInstruction(
                    "",
                    "break;",
                    within_inames=context.inames,
                    predicates=context.predicates,
                    id=context.name_gen(LOOPY_INSN_PREFIX),
                )
            ],
            [],
        )

    # pylint-disable-reason: `expr` has to be present even though unused
    # since all the members of the mapper must have the same API
    def map_continue_stmt(
        self,
        expr: cindex.CursorKind.CONTINUE_STMT,  # pylint: disable=W0613
        context: CToLoopyMapperContext,
    ) -> CToLoopyMapperAccumulator:
        """Maps a C continue statement."""
        return CToLoopyMapperAccumulator(
            [],
            [
                lp.CInstruction(
                    "",
                    "continue;",
                    within_inames=context.inames,
                    predicates=context.predicates,
                    id=context.name_gen(LOOPY_INSN_PREFIX),
                )
            ],
            [],
        )


class CKernel:
    """Store the plain C kernel."""

    knl_name: str
    knl_body: cindex.Cursor
    var_to_decl: Dict[str, cindex.Cursor]

    def __init__(self, function: cindex.CursorKind.FUNCTION_DECL):
        self.knl_name = function.spelling
        self.var_to_decl = {}

        (*args, self.knl_body) = function.get_children()
        for arg in args:
            if arg.kind == cindex.CursorKind.PARM_DECL:
                self.var_to_decl[arg.spelling] = arg

    def get_knl_args(self) -> List[lp.KernelArgument]:
        """Return Loopy kernel arguments for C function argument."""
        knl_args = []
        for arg in self.var_to_decl.values():
            dtype = _get_dtype_from_decl_type(arg.type)
            if arg.type.kind in _ARRAY_TYPES_W_PTR:
                knl_args.append(
                    lp.ArrayArg(
                        arg.spelling,
                        dtype=dtype,
                        address_space=AddressSpace.GLOBAL,
                    )
                )
            elif isinstance(arg.type.kind, cindex.TypeKind):
                knl_args.append(lp.ValueArg(arg.spelling, dtype=dtype))
            else:
                raise NotImplementedError(
                    f"get_knl_args: {arg.type.kind} is of invalid type:"
                    f" {dtype}."
                )
        return knl_args

    def get_value_types(self) -> dict:
        """Return value type declarations."""
        return [
            k
            for k, v in self.var_to_decl.items()
            if v.type.kind not in _ARRAY_TYPES_W_PTR
        ]

    def get_knl_body(self) -> cindex.Cursor:
        """Return the C kernel body."""
        return self.knl_body

    def get_knl_name(self) -> str:
        """Return the C kernel name."""
        return self.knl_name


def c_to_loopy(c_str: str, backend: str) -> lp.translation_unit.TranslationUnit:
    """Returns a Loopy kernel for a C loop band."""
    fname = hashlib.sha256(c_str.encode("utf-8")).hexdigest() + ".c"
    tunit = cindex.Index.create().parse(fname, unsaved_files=[(fname, c_str)])

    # Check for syntax errors in parsed C kernel
    errors = [diagnostic.spelling for diagnostic in tunit.diagnostics]
    if len(errors) > 0:
        raise SyntaxError(f"Failed to parse C source due to errors: {errors}")

    # Init `var_to_decl` based on function parameters
    (function,) = tunit.cursor.get_children()
    c_knl = CKernel(function)

    # Map C for loop to loopy kernel
    acc = CToLoopyMapper()(
        c_knl.get_knl_body(),
        CToLoopyMapperContext(
            frozenset(),
            c_knl.get_value_types(),
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
        c_knl.get_knl_args() + acc.kernel_data + [...],
        lang_version=LOOPY_LANG_VERSION,
        name=c_knl.get_knl_name(),
        seq_dependencies=True,
        target=_BACKEND_TO_TARGET[backend],
    )

    return knl


def get_wrapper(backend: str) -> BaseKernelWrapper:
    """Returns the wrapper for a given backend."""
    wrapper = _BACKEND_TO_WRAPPER.get(backend)
    return wrapper if wrapper is not None else BaseKernelWrapper()


def get_knl_src(knl: lp.translation_unit.TranslationUnit, backend: str) -> str:
    """Returns the kernel source for a given backend."""
    return get_wrapper(backend).get_src(knl)


def get_knl_name(knl: lp.translation_unit.TranslationUnit, backend: str) -> str:
    """Returns the kernel name for a given backend."""
    return get_wrapper(backend).get_entry_point(knl)


if __name__ == "__main__":
    import os
    import sys

    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    KNL_STR = """
          void foo(int *a, int N) {
            for (int i = 0; i < N; i++) {
              int t = 1;
              for (int j = 0; j < 10; j++) {
                t = N < 5 ? i : (i + 3);
                if (j == 5 || !(j == 6))
                  continue;
              }
              a[i] = t;
            }
          }
          """
    BACKEND = "cuda"
    lp_knl = c_to_loopy(KNL_STR, BACKEND)
    print(get_knl_name(lp_knl, BACKEND))
    print(get_knl_src(lp_knl, BACKEND))
