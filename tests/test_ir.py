"""Unit tests for Glassbox shared contracts and IR AST."""

from __future__ import annotations

import math

import numpy as np
import pytest
from pydantic import ValidationError

from glassbox.ir import (
    BinaryExpr,
    ClampExpr,
    ConstExpr,
    DecompiledProgram,
    DecompileResult,
    GuardNode,
    LinearGuard,
    RegimeLeaf,
    UnaryExpr,
    VarExpr,
    eval_expr,
    is_expr_linear,
)
from glassbox.types import (
    CEGISIteration,
    DecompileConfig,
    DomainSpec,
    LayerInfo,
    ModelMetadata,
    TaskType,
    VerificationCertificate,
    VerificationStatus,
)
from glassbox.types import (
    DecompileResult as TypesDecompileResult,
)


def _sample_domain(task_type: TaskType = TaskType.REGRESSION) -> DomainSpec:
    return DomainSpec(
        feature_names=["x0", "x1"],
        lower_bounds=[-2.0, -1.0],
        upper_bounds=[2.0, 3.0],
        output_names=["y0", "y1"],
        task_type=task_type,
        class_names=["c0", "c1"] if task_type == TaskType.CLASSIFICATION else [],
    )


def _sample_program(task_type: TaskType = TaskType.REGRESSION) -> DecompiledProgram:
    domain = _sample_domain(task_type=task_type)
    leaf0 = RegimeLeaf(
        regime_id=0,
        name="left_low",
        description="x0 <= 0 and x1 < 1",
        output_exprs=[
            BinaryExpr(
                op="add",
                left=BinaryExpr(op="mul", left=ConstExpr(value=2.0), right=VarExpr(index=0, name="x0")),
                right=ConstExpr(value=1.0),
            ),
            ClampExpr(operand=VarExpr(index=1, name="x1"), low=0.0, high=2.0),
        ],
        class_id=0 if task_type == TaskType.CLASSIFICATION else None,
        class_name="c0" if task_type == TaskType.CLASSIFICATION else None,
        sample_count=50,
        local_error=0.01,
    )
    leaf1 = RegimeLeaf(
        regime_id=1,
        name="left_high",
        description="x0 <= 0 and x1 >= 1",
        output_exprs=[
            UnaryExpr(op="relu", operand=VarExpr(index=1, name="x1")),
            UnaryExpr(op="abs", operand=VarExpr(index=0, name="x0")),
        ],
        class_id=1 if task_type == TaskType.CLASSIFICATION else None,
        class_name="c1" if task_type == TaskType.CLASSIFICATION else None,
        sample_count=30,
        local_error=0.02,
    )
    leaf2 = RegimeLeaf(
        regime_id=2,
        name="right",
        description="x0 > 0",
        output_exprs=[
            BinaryExpr(
                op="min",
                left=VarExpr(index=0, name="x0"),
                right=VarExpr(index=1, name="x1"),
            ),
            BinaryExpr(
                op="max",
                left=UnaryExpr(op="neg", operand=VarExpr(index=0, name="x0")),
                right=BinaryExpr(op="div", left=VarExpr(index=1, name="x1"), right=ConstExpr(value=2.0)),
            ),
        ],
        class_id=None,
        sample_count=40,
        local_error=0.005,
    )
    inner_guard = GuardNode(
        guard=LinearGuard(
            weights=[0.0, 1.0],
            bias=-1.0,
            feature_names=["x0", "x1"],
            op="<",
        ),
        true_branch=leaf0,
        false_branch=leaf1,
    )
    root_guard = GuardNode(
        guard=LinearGuard(
            weights=[1.0, 0.0],
            bias=0.0,
            feature_names=["x0", "x1"],
            op="<=",
        ),
        true_branch=inner_guard,
        false_branch=leaf2,
    )
    return DecompiledProgram(
        name="test_prog",
        task_type=task_type,
        domain=domain,
        root=root_guard,
    )


def test_domain_spec_validation() -> None:
    dom = _sample_domain(TaskType.CLASSIFICATION)
    assert dom.input_dim == 2
    assert dom.output_dim == 2

    with pytest.raises(ValidationError):
        DomainSpec(
            feature_names=[],
            lower_bounds=[],
            upper_bounds=[],
            output_names=["y0"],
            task_type=TaskType.REGRESSION,
        )

    with pytest.raises(ValidationError):
        DomainSpec(
            feature_names=["x0"],
            lower_bounds=[0.0],
            upper_bounds=[1.0],
            output_names=[],
            task_type=TaskType.REGRESSION,
        )

    with pytest.raises(ValidationError):
        DomainSpec(
            feature_names=["x0", "x1"],
            lower_bounds=[0.0],
            upper_bounds=[1.0, 2.0],
            output_names=["y0"],
            task_type=TaskType.REGRESSION,
        )

    with pytest.raises(ValidationError):
        DomainSpec(
            feature_names=["x0"],
            lower_bounds=[1.0],
            upper_bounds=[1.0],
            output_names=["y0"],
            task_type=TaskType.REGRESSION,
        )

    with pytest.raises(ValidationError):
        DomainSpec(
            feature_names=["x0"],
            lower_bounds=[2.0],
            upper_bounds=[1.0],
            output_names=["y0"],
            task_type=TaskType.REGRESSION,
        )

    with pytest.raises(ValidationError):
        DomainSpec(
            feature_names=["x0"],
            lower_bounds=[float("nan")],
            upper_bounds=[1.0],
            output_names=["y0"],
            task_type=TaskType.REGRESSION,
        )


def test_linear_guard_ops_and_formatting() -> None:
    with pytest.raises(ValidationError):
        LinearGuard(weights=[1.0], bias=0.0, feature_names=["x0", "x1"])

    g_le = LinearGuard(weights=[1.0, -2.0, 0.0], bias=0.5, feature_names=["a", "b", "c"], op="<=")
    assert g_le.margin([1.0, 0.75, 99.0]) == pytest.approx(0.0)
    assert g_le.evaluate([1.0, 0.75, 99.0]) is True
    assert g_le.evaluate([2.0, 0.0, 0.0]) is False

    g_lt = LinearGuard(weights=[1.0, 0.0], bias=0.0, feature_names=["a", "b"], op="<")
    assert g_lt.evaluate([0.0, 0.0]) is False
    assert g_lt.evaluate([-0.1, 0.0]) is True

    g_ge = LinearGuard(weights=[1.0, 0.0], bias=0.0, feature_names=["a", "b"], op=">=")
    assert g_ge.evaluate([0.0, 0.0]) is True
    assert g_ge.evaluate([-0.1, 0.0]) is False

    g_gt = LinearGuard(weights=[1.0, 0.0], bias=0.0, feature_names=["a", "b"], op=">")
    assert g_gt.evaluate([0.0, 0.0]) is False
    assert g_gt.evaluate([0.1, 0.0]) is True

    assert g_le.active_terms() == [(0, "a", 1.0), (1, "b", -2.0)]
    readable = g_le.to_readable(precision=2)
    assert readable == "1.00*a - 2.00*b + 0.50 <= 0"


def test_eval_expr_all_ops() -> None:
    x = [0.5, -1.5]
    x0 = VarExpr(index=0, name="x0")
    x1 = VarExpr(index=1, name="x1")

    assert eval_expr(ConstExpr(value=3.5), x) == pytest.approx(3.5)
    assert eval_expr(UnaryExpr(op="neg", operand=x0), x) == pytest.approx(-0.5)
    assert eval_expr(UnaryExpr(op="abs", operand=x1), x) == pytest.approx(1.5)
    assert eval_expr(UnaryExpr(op="relu", operand=x1), x) == pytest.approx(0.0)
    assert eval_expr(UnaryExpr(op="relu", operand=x0), x) == pytest.approx(0.5)
    assert eval_expr(UnaryExpr(op="tanh", operand=x0), x) == pytest.approx(math.tanh(0.5))
    assert eval_expr(UnaryExpr(op="sin", operand=x0), x) == pytest.approx(math.sin(0.5))
    assert eval_expr(UnaryExpr(op="cos", operand=x0), x) == pytest.approx(math.cos(0.5))
    assert eval_expr(BinaryExpr(op="sub", left=x0, right=x1), x) == pytest.approx(2.0)
    assert eval_expr(ClampExpr(operand=x1, low=-1.0, high=1.0), x) == pytest.approx(-1.0)


def test_is_expr_linear_and_program_linearity() -> None:
    x0 = VarExpr(index=0, name="x0")
    x1 = VarExpr(index=1, name="x1")
    c2 = ConstExpr(value=2.0)
    c0 = ConstExpr(value=0.0)

    assert is_expr_linear(x0) is True
    assert is_expr_linear(c2) is True
    assert is_expr_linear(UnaryExpr(op="relu", operand=x0)) is True
    assert is_expr_linear(UnaryExpr(op="abs", operand=x0)) is True
    assert is_expr_linear(UnaryExpr(op="neg", operand=x0)) is True
    assert is_expr_linear(ClampExpr(operand=x0, low=-1.0, high=1.0)) is True
    assert is_expr_linear(BinaryExpr(op="add", left=x0, right=x1)) is True
    assert is_expr_linear(BinaryExpr(op="sub", left=x0, right=x1)) is True
    assert is_expr_linear(BinaryExpr(op="min", left=x0, right=x1)) is True
    assert is_expr_linear(BinaryExpr(op="max", left=x0, right=x1)) is True
    assert is_expr_linear(BinaryExpr(op="mul", left=c2, right=x0)) is True
    assert is_expr_linear(BinaryExpr(op="mul", left=x0, right=c2)) is True
    assert is_expr_linear(BinaryExpr(op="div", left=x0, right=c2)) is True

    # Non-linear expressions
    assert is_expr_linear(UnaryExpr(op="tanh", operand=x0)) is False
    assert is_expr_linear(UnaryExpr(op="sin", operand=x0)) is False
    assert is_expr_linear(UnaryExpr(op="cos", operand=x0)) is False
    assert is_expr_linear(BinaryExpr(op="mul", left=x0, right=x1)) is False
    assert is_expr_linear(BinaryExpr(op="div", left=x0, right=x1)) is False
    assert is_expr_linear(BinaryExpr(op="div", left=x0, right=c0)) is False

    prog = _sample_program()
    assert prog.is_piecewise_linear() is True

    nonlinear_prog = prog.model_copy(deep=True)
    assert isinstance(nonlinear_prog.root, GuardNode)
    assert isinstance(nonlinear_prog.root.false_branch, RegimeLeaf)
    nonlinear_prog.root.false_branch.output_exprs[0] = UnaryExpr(op="tanh", operand=x0)
    assert nonlinear_prog.is_piecewise_linear() is False


def test_program_evaluation_metrics_and_classes() -> None:
    prog = _sample_program(TaskType.CLASSIFICATION)

    out0, rid0 = prog.evaluate_one([-1.0, 0.5])
    assert rid0 == 0
    assert out0 == pytest.approx([-1.0, 0.5])

    out1, rid1 = prog.evaluate_one([0.0, 1.0])
    assert rid1 == 1
    assert out1 == pytest.approx([1.0, 0.0])

    out2, rid2 = prog.evaluate_one([1.0, 2.0])
    assert rid2 == 2
    assert out2 == pytest.approx([1.0, 1.0])

    X = np.array([[-1.0, 0.5], [0.0, 1.0], [1.0, 2.0]])
    preds = prog.predict(X)
    assert preds.shape == (3, 2)
    assert np.allclose(preds, np.array([[-1.0, 0.5], [1.0, 0.0], [1.0, 1.0]]))

    preds2, regimes = prog.predict_with_regimes(X)
    assert np.allclose(preds2, preds)
    assert list(regimes) == [0, 1, 2]

    classes = prog.predict_classes(X)
    # leaf0 class_id=0, leaf1 class_id=1, leaf2 class_id=None -> argmax([1.0, 1.0]) == 0
    assert list(classes) == [0, 1, 0]

    # Empty batch
    empty_preds, empty_regimes = prog.predict_with_regimes(np.empty((0, 2)))
    assert empty_preds.shape == (0, 2)
    assert empty_regimes.shape == (0,)
    assert prog.predict_classes(np.empty((0, 2))).shape == (0,)

    with pytest.raises(ValueError):
        prog.predict(np.array([1.0, 2.0]))

    assert [leaf.regime_id for leaf in prog.leaves()] == [0, 1, 2]
    assert prog.max_depth() == 2
    assert prog.cyclomatic_complexity() == 3
    # 2 guard nodes + 3 leaf nodes + expr nodes:
    # leaf0: mul(3) + add(+2=5) + clamp(2) => 7 expr nodes
    # leaf1: relu(2) + abs(2) => 4 expr nodes
    # leaf2: min(3) + neg(2) + div(3) + max(+1=6) => 9 expr nodes
    # total = 5 tree nodes + 20 expr nodes = 25
    assert prog.ast_node_count() == 25


def test_serialization_and_sha256_determinism() -> None:
    prog1 = _sample_program(TaskType.REGRESSION)
    prog2 = _sample_program(TaskType.REGRESSION)
    assert prog1.sha256() == prog2.sha256()

    raw_json = prog1.model_dump_json()
    restored = DecompiledProgram.model_validate_json(raw_json)
    assert restored.sha256() == prog1.sha256()
    assert restored == prog1

    modified = prog1.model_copy(deep=True)
    modified.name = "modified_prog"
    assert modified.sha256() != prog1.sha256()


def test_decompile_result_exports_and_roundtrip() -> None:
    assert DecompileResult is TypesDecompileResult

    prog = _sample_program(TaskType.REGRESSION)
    dom = prog.domain
    meta = ModelMetadata(
        name="mlp_test",
        path="models/mlp_test.onnx",
        sha256="abc123",
        size_bytes=1024,
        input_dim=2,
        output_dim=2,
        total_params=16,
        is_piecewise_linear=True,
        activations=["relu"],
        layers=[
            LayerInfo(
                name="fc1",
                op_type="Gemm",
                in_features=2,
                out_features=2,
                activation="relu",
                param_count=6,
            )
        ],
        domain=dom,
    )
    cert = VerificationCertificate(
        created_at="2026-11-23T00:00:00Z",
        model_name="mlp_test",
        model_sha256="abc123",
        program_sha256=prog.sha256(),
        status=VerificationStatus.PROVEN_SMT,
        tolerance_eps=0.05,
        classification_margin=0.05,
        domain=dom,
        agreement_rate=1.0,
        max_abs_error=0.01,
        mean_abs_error=0.005,
        smt_checked=True,
        smt_proven=True,
        smt_timeout_s=4.0,
        total_counterexamples_found=0,
        cegis_iterations=[
            CEGISIteration(
                iteration=1,
                agreement_rate=1.0,
                max_abs_error=0.01,
                mean_abs_error=0.005,
                rmse=0.006,
                regime_count=3,
                ast_nodes=prog.ast_node_count(),
                status=VerificationStatus.PROVEN_SMT,
                counterexamples_found=0,
            )
        ],
    )
    cfg = DecompileConfig()
    assert cfg.regime_method == "oblique_tree"

    res = DecompileResult(
        run_id="run-001",
        metadata=meta,
        program=prog,
        certificate=cert,
    )
    roundtrip = DecompileResult.model_validate_json(res.model_dump_json())
    assert roundtrip.run_id == "run-001"
    assert roundtrip.program.sha256() == prog.sha256()
