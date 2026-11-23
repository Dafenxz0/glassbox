"""Intermediate representation (IR) AST, guards, and decompiled programs."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import BaseModel, Field, model_validator

from glassbox.types import (
    BenchmarkEntry,
    Counterexample,
    DecisionExplanation,
    DomainSpec,
    ModelMetadata,
    ParetoCandidate,
    TaskType,
    VerificationCertificate,
)


class VarExpr(BaseModel):
    """Reference to an input feature variable by index."""

    kind: Literal["var"] = "var"
    index: int
    name: str


class ConstExpr(BaseModel):
    """Scalar floating-point constant expression."""

    kind: Literal["const"] = "const"
    value: float


class UnaryExpr(BaseModel):
    """Unary mathematical operation over an expression."""

    kind: Literal["unary"] = "unary"
    op: Literal["neg", "abs", "relu", "tanh", "sin", "cos"]
    operand: Expr


class BinaryExpr(BaseModel):
    """Binary mathematical operation over two expressions."""

    kind: Literal["binary"] = "binary"
    op: Literal["add", "sub", "mul", "div", "min", "max"]
    left: Expr
    right: Expr


class ClampExpr(BaseModel):
    """Clamping expression restricting an operand to [low, high]."""

    kind: Literal["clamp"] = "clamp"
    operand: Expr
    low: float
    high: float


Expr = Annotated[
    VarExpr | ConstExpr | UnaryExpr | BinaryExpr | ClampExpr,
    Field(discriminator="kind"),
]

UnaryExpr.model_rebuild()
BinaryExpr.model_rebuild()
ClampExpr.model_rebuild()


def eval_expr(expr: Expr, x: Sequence[float]) -> float:
    """Evaluate an IR expression at concrete input vector x."""
    if isinstance(expr, VarExpr):
        return float(x[expr.index])
    if isinstance(expr, ConstExpr):
        return float(expr.value)
    if isinstance(expr, UnaryExpr):
        v = eval_expr(expr.operand, x)
        if expr.op == "neg":
            return -v
        if expr.op == "abs":
            return abs(v)
        if expr.op == "relu":
            return max(0.0, v)
        if expr.op == "tanh":
            return math.tanh(v)
        if expr.op == "sin":
            return math.sin(v)
        if expr.op == "cos":
            return math.cos(v)
        raise ValueError(f"Unsupported unary op: {expr.op}")
    if isinstance(expr, BinaryExpr):
        lv = eval_expr(expr.left, x)
        rv = eval_expr(expr.right, x)
        if expr.op == "add":
            return lv + rv
        if expr.op == "sub":
            return lv - rv
        if expr.op == "mul":
            return lv * rv
        if expr.op == "div":
            return lv / rv
        if expr.op == "min":
            return min(rv, lv)
        if expr.op == "max":
            return max(rv, lv)
        raise ValueError(f"Unsupported binary op: {expr.op}")
    if isinstance(expr, ClampExpr):
        v = eval_expr(expr.operand, x)
        if v < expr.low:
            return float(expr.low)
        if v > expr.high:
            return float(expr.high)
        return v
    raise TypeError(f"Unsupported expr type: {type(expr)!r}")


def _eval_if_const(expr: Expr) -> float | None:
    """Return constant value if expr contains no variables and only arithmetic ops, else None."""
    if isinstance(expr, ConstExpr):
        return float(expr.value)
    if isinstance(expr, VarExpr):
        return None
    if isinstance(expr, UnaryExpr):
        if expr.op in ("tanh", "sin", "cos"):
            return None
        v = _eval_if_const(expr.operand)
        if v is None:
            return None
        return eval_expr(UnaryExpr(op=expr.op, operand=ConstExpr(value=v)), ())
    if isinstance(expr, BinaryExpr):
        lv = _eval_if_const(expr.left)
        rv = _eval_if_const(expr.right)
        if lv is None or rv is None:
            return None
        if expr.op == "div" and rv == 0.0:
            return None
        return eval_expr(
            BinaryExpr(op=expr.op, left=ConstExpr(value=lv), right=ConstExpr(value=rv)),
            (),
        )
    if isinstance(expr, ClampExpr):
        v = _eval_if_const(expr.operand)
        if v is None:
            return None
        return eval_expr(ClampExpr(operand=ConstExpr(value=v), low=expr.low, high=expr.high), ())
    return None


def is_expr_linear(expr: Expr) -> bool:
    """Return True if expr is piecewise linear (expressible in Z3 QF_LRA).

    Piecewise-linear operations include variables, constants, negation, addition,
    subtraction, multiplication by a constant, division by a nonzero constant,
    and piecewise-linear primitives (relu, abs, min, max, clamp). Transcendental
    functions (tanh, sin, cos) and variable-variable products/divisions are not.
    """
    if isinstance(expr, (VarExpr, ConstExpr)):
        return True
    if isinstance(expr, UnaryExpr):
        if expr.op in ("neg", "abs", "relu"):
            return is_expr_linear(expr.operand)
        return False
    if isinstance(expr, BinaryExpr):
        if expr.op in ("add", "sub", "min", "max"):
            return is_expr_linear(expr.left) and is_expr_linear(expr.right)
        if expr.op == "mul":
            if not (is_expr_linear(expr.left) and is_expr_linear(expr.right)):
                return False
            return _eval_if_const(expr.left) is not None or _eval_if_const(expr.right) is not None
        if expr.op == "div":
            rv = _eval_if_const(expr.right)
            return is_expr_linear(expr.left) and rv is not None and rv != 0.0
        return False
    if isinstance(expr, ClampExpr):
        return is_expr_linear(expr.operand)
    return False


def _count_expr_nodes(expr: Expr) -> int:
    if isinstance(expr, (VarExpr, ConstExpr)):
        return 1
    if isinstance(expr, (UnaryExpr, ClampExpr)):
        return 1 + _count_expr_nodes(expr.operand)
    if isinstance(expr, BinaryExpr):
        return 1 + _count_expr_nodes(expr.left) + _count_expr_nodes(expr.right)
    return 1


class LinearGuard(BaseModel):
    """Hyperplane guard condition of the form (w . x + b) <op> 0."""

    weights: list[float]
    bias: float
    feature_names: list[str]
    op: Literal["<=", "<", ">=", ">"] = "<="

    @model_validator(mode="after")
    def _validate_lengths(self) -> LinearGuard:
        if len(self.weights) != len(self.feature_names):
            raise ValueError("weights and feature_names must have equal lengths")
        return self

    def margin(self, x: Sequence[float]) -> float:
        """Compute signed affine value w . x + b."""
        if len(x) != len(self.weights):
            raise ValueError(f"Expected {len(self.weights)} features, got {len(x)}")
        return sum(float(w) * float(xi) for w, xi in zip(self.weights, x)) + float(self.bias)

    def evaluate(self, x: Sequence[float]) -> bool:
        """Evaluate whether x satisfies the guard predicate."""
        m = self.margin(x)
        if self.op == "<=":
            return m <= 0.0
        if self.op == "<":
            return m < 0.0
        if self.op == ">=":
            return m >= 0.0
        if self.op == ">":
            return m > 0.0
        raise ValueError(f"Unsupported guard op: {self.op}")

    def active_terms(self, tol: float = 1e-9) -> list[tuple[int, str, float]]:
        """Return (index, name, weight) tuples for non-negligible weights."""
        return [
            (i, name, float(w))
            for i, (name, w) in enumerate(zip(self.feature_names, self.weights))
            if abs(w) > tol
        ]

    def to_readable(self, precision: int = 4) -> str:
        """Format guard as a human-readable linear inequality string."""
        terms: list[str] = []
        for _, name, w in self.active_terms():
            abs_w = f"{abs(w):.{precision}f}"
            if not terms:
                sign = "-" if w < 0 else ""
                terms.append(f"{sign}{abs_w}*{name}")
            else:
                sign = " - " if w < 0 else " + "
                terms.append(f"{sign}{abs_w}*{name}")
        lhs = "".join(terms) if terms else f"{0.0:.{precision}f}"
        if abs(self.bias) > 1e-12 or not terms:
            if terms:
                b_sign = " - " if self.bias < 0 else " + "
                lhs += f"{b_sign}{abs(self.bias):.{precision}f}"
            else:
                lhs = f"{self.bias:.{precision}f}"
        return f"{lhs} {self.op} 0"


class RegimeLeaf(BaseModel):
    """Leaf node representing a single local regime and its symbolic output expressions."""

    node_type: Literal["leaf"] = "leaf"
    regime_id: int
    name: str
    description: str
    output_exprs: list[Expr]
    class_id: int | None = None
    class_name: str | None = None
    sample_count: int = 0
    local_error: float = 0.0


class GuardNode(BaseModel):
    """Internal decision-tree node splitting on a LinearGuard."""

    node_type: Literal["guard"] = "guard"
    guard: LinearGuard
    true_branch: RegimeNode
    false_branch: RegimeNode


RegimeNode = Annotated[GuardNode | RegimeLeaf, Field(discriminator="node_type")]
GuardNode.model_rebuild()


class DecompiledProgram(BaseModel):
    """Complete piecewise symbolic program decompiled from a neural network."""

    name: str
    task_type: TaskType
    domain: DomainSpec
    root: RegimeNode
    is_linear_ir: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    def _route_leaf(self, x: Sequence[float]) -> RegimeLeaf:
        node = self.root
        while isinstance(node, GuardNode):
            node = node.true_branch if node.guard.evaluate(x) else node.false_branch
        return node

    def trace_one(
        self, x: Sequence[float]
    ) -> tuple[list[float], RegimeLeaf, list[dict[str, Any]]]:
        """Trace decision-tree routing and evaluate output expressions for a single input."""
        node = self.root
        guard_trace: list[dict[str, Any]] = []
        while isinstance(node, GuardNode):
            margin = node.guard.margin(x)
            satisfied = node.guard.evaluate(x)
            guard_trace.append(
                {
                    "expression": node.guard.to_readable(),
                    "margin": margin,
                    "satisfied": satisfied,
                    "branch": "true" if satisfied else "false",
                }
            )
            node = node.true_branch if satisfied else node.false_branch
        outputs = [eval_expr(expr, x) for expr in node.output_exprs]
        return outputs, node, guard_trace

    def evaluate_one(self, x: Sequence[float]) -> tuple[list[float], int]:
        """Evaluate the program on a single input, returning (outputs, regime_id)."""
        leaf = self._route_leaf(x)
        outputs = [eval_expr(expr, x) for expr in leaf.output_exprs]
        return outputs, leaf.regime_id

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Evaluate the program over a 2D batch of inputs."""
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D input array, got shape {arr.shape}")
        rows = [self.evaluate_one(row)[0] for row in arr]
        if not rows:
            return np.empty((0, self.domain.output_dim), dtype=np.float64)
        return np.asarray(rows, dtype=np.float64)

    def predict_with_regimes(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate outputs and active regime IDs for a 2D batch of inputs."""
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D input array, got shape {arr.shape}")
        if len(arr) == 0:
            return (
                np.empty((0, self.domain.output_dim), dtype=np.float64),
                np.empty((0,), dtype=np.int64),
            )
        outputs: list[list[float]] = []
        regimes: list[int] = []
        for row in arr:
            out, rid = self.evaluate_one(row)
            outputs.append(out)
            regimes.append(rid)
        return np.asarray(outputs, dtype=np.float64), np.asarray(regimes, dtype=np.int64)

    def predict_classes(self, X: np.ndarray) -> np.ndarray:
        """Predict class indices using leaf.class_id when set, else argmax(outputs)."""
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D input array, got shape {arr.shape}")
        if len(arr) == 0:
            return np.empty((0,), dtype=np.int64)
        classes: list[int] = []
        for row in arr:
            leaf = self._route_leaf(row)
            if leaf.class_id is not None:
                classes.append(int(leaf.class_id))
            else:
                outs = [eval_expr(expr, row) for expr in leaf.output_exprs]
                classes.append(int(np.argmax(outs)))
        return np.asarray(classes, dtype=np.int64)

    def leaves(self) -> list[RegimeLeaf]:
        """Return all RegimeLeaf nodes in depth-first true-before-false order."""
        result: list[RegimeLeaf] = []

        def _walk(node: RegimeNode) -> None:
            if isinstance(node, RegimeLeaf):
                result.append(node)
            else:
                _walk(node.true_branch)
                _walk(node.false_branch)

        _walk(self.root)
        return result

    def ast_node_count(self) -> int:
        """Return total count of decision-tree nodes and leaf expression AST nodes."""

        def _walk(node: RegimeNode) -> int:
            if isinstance(node, RegimeLeaf):
                return 1 + sum(_count_expr_nodes(e) for e in node.output_exprs)
            return 1 + _walk(node.true_branch) + _walk(node.false_branch)

        return _walk(self.root)

    def max_depth(self) -> int:
        """Return maximum number of guard decisions from root to any leaf."""

        def _depth(node: RegimeNode) -> int:
            if isinstance(node, RegimeLeaf):
                return 0
            return 1 + max(_depth(node.true_branch), _depth(node.false_branch))

        return _depth(self.root)

    def cyclomatic_complexity(self) -> int:
        """Return McCabe cyclomatic complexity (number of guards + 1 == leaf count)."""
        return len(self.leaves())

    def sha256(self) -> str:
        """Return deterministic SHA-256 hex digest of canonical JSON representation."""
        canonical = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def is_piecewise_linear(self) -> bool:
        """Return True iff every output expression across all leaves is piecewise linear."""
        return all(
            is_expr_linear(expr)
            for leaf in self.leaves()
            for expr in leaf.output_exprs
        )


class DecompileResult(BaseModel):
    """End-to-end result bundle produced by Glassbox decompilation."""

    run_id: str
    metadata: ModelMetadata
    program: DecompiledProgram
    certificate: VerificationCertificate
    counterexamples: list[Counterexample] = Field(default_factory=list)
    code: dict[str, str] = Field(default_factory=dict)
    benchmarks: list[BenchmarkEntry] = Field(default_factory=list)
    pareto_frontier: list[ParetoCandidate] = Field(default_factory=list)
    regime_neuron_profiles: dict[int, dict[str, Any]] = Field(default_factory=dict)
    sample_explanations: list[DecisionExplanation] = Field(default_factory=list)


__all__ = [
    "BinaryExpr",
    "ClampExpr",
    "ConstExpr",
    "DecompileResult",
    "DecompiledProgram",
    "Expr",
    "GuardNode",
    "LinearGuard",
    "RegimeLeaf",
    "RegimeNode",
    "UnaryExpr",
    "VarExpr",
    "eval_expr",
    "is_expr_linear",
]
