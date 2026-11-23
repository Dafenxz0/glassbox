# CEGIS Synthesis, Exact Rational SMT & Two-Tier Assurance

GlassBox separates **inductive synthesis** from **formal verification** in a Counterexample-Guided Inductive Synthesis (CEGIS) loop, and separates **Real-Arithmetic SMT Proofs** from **Numerical Deployment Assurance**.

## Two-Tier Assurance Model

Because ONNX models store weights in IEEE-754 `float32` and execute on finite-precision hardware, while SMT solvers over linear real arithmetic (`QF_LRA`) reason over mathematical reals $\mathbb{R}$ ($\mathbb{Q}$), GlassBox reports two distinct, complementary records in every `VerificationCertificate`:

1. **`RealArithmeticProof` (Mathematical Symbolic Equivalence)**:
   - Converts every IEEE-754 float weight, bias, domain bound, guard coefficient, and IR constant into its **exact rational value** via `fractions.Fraction(float(v))` and encodes it as an exact Z3 rational `z3.Q(numerator, denominator)`.
   - Uses **zero floating-point truncation** (no 8-significant-figure string rounding, no dropping of small non-zero weights) and **zero unverified floating-point LP cuts** (only sound outward-rounded exact `Fraction` Interval Bound Propagation over each leaf polyhedron's bounding box).
   - Certifies whether the synthesized `DecompiledProgram` is mathematically equivalent to the rational MLP defined by the ONNX weights across the continuous domain $[l, u] \subset \mathbb{R}^d$.
2. **`NumericalDeploymentAssurance` (Finite-Precision Execution Fidelity)**:
   - Audits the finite-precision gap between `float32` ONNXRuntime execution, `float64` IR evaluation, and compiled native C99 (`double`) / zero-dependency Python outputs across deterministic domain test suites (`ir_vs_onnx_max_diff`, `c99_vs_ir_max_diff`, `python_vs_ir_max_diff`).

---

## Verification Statuses at a Glance

| `VerificationStatus` | Tier | Scope | Exact Meaning |
| :--- | :---: | :---: | :--- |
| **`PROVEN_REAL_GLOBAL`** | Tier A (Z3 `QF_LRA`) | `global` | Exact rational Z3 returned `unsat` across **all** regime leaves on the entire `DomainSpec` hyperrectangle (`||NN(x) - P(x)||_inf <= eps` for regression, or identical argmax class with `classification_margin = 0.0`). |
| **`PROVEN_REAL_MARGIN`** | Tier A (Z3 `QF_LRA`) | `margin_conditioned` | Exact rational Z3 returned `unsat` across all inputs where the neural network's top-2 logit margin satisfies $z_{c^*} - z_j \ge \gamma$ (`classification_margin > 0.0`), excluding ambiguous near-zero tie zones. |
| **`TIMEOUT`** | Tier A (Z3 `QF_LRA`) | `none` | Z3 exceeded `smt_timeout_s` on one or more leaves without finding a counterexample. Never collapsed into `BOUNDED_EMPIRICAL`. |
| **`UNKNOWN`** | Tier A (Z3 `QF_LRA`) | `none` | Z3 returned `unknown` (non-timeout) on one or more leaves. |
| **`BOUNDED_EMPIRICAL`** | Tier B (Adversarial) | `none` | Smooth network (`Tanh`, `Sigmoid`) or `--no-smt` run: 4096-point Sobol + corner + guard-boundary + multi-start Nelder-Mead search found zero violations above `eps`. Explicitly **not** a formal proof. |
| **`COUNTEREXAMPLE_FOUND`** | Tier A or Tier B | `none` | A concrete witness $x^* \in [l, u]$ where the program violates `eps` or class agreement was found and logged. |
| **`UNVERIFIED`** | None | `none` | Verification has not been executed. |

---

## Tier A: Exact Rational `QF_LRA` Encoding (ReLU MLPs)

A feed-forward ReLU MLP on a bounded real box $x \in [l, u] \subset \mathbb{R}^d$ is piecewise affine. Both the network and a linear `DecompiledProgram` (`is_piecewise_linear() == True`) are expressible in **Quantifier-Free Linear Real Arithmetic (`QF_LRA`)**.

### 1. Exact Rational Lifting & Sound Outward IBP
For every scalar parameter $v \in \mathbb{F}_{64}$, GlassBox lifts $v$ without rounding:
$$q(v) = \text{Fraction}(\text{float}(v)) \in \mathbb{Q}$$

For each regime leaf $r$ with guard conjunction $G_r = \{w_m^\top x + b_m \le 0 \text{ or } \ge 0\}$, GlassBox propagates exact rational intervals $[L_{k,j}, U_{k,j}] \subset \mathbb{Q}$ layer-by-layer using outward-safe bounds:
- If $L_{k,j} \ge 0$, neuron $(k,j)$ is **provably active** ($h_{k,j} = z_{k,j}$).
- If $U_{k,j} \le 0$, neuron $(k,j)$ is **provably inactive** ($h_{k,j} = 0$).
- If $L_{k,j} < 0 < U_{k,j}$, neuron $(k,j)$ is encoded exactly via `z3.If(z >= 0, z, 0)` along with sound bounding inequalities $h_{k,j} \ge 0$, $h_{k,j} \ge z_{k,j}$, and $h_{k,j} \le \frac{U_{k,j}}{U_{k,j} - L_{k,j}}(z_{k,j} - L_{k,j})$.

### 2. Per-Leaf Equivalence Negation Query
Rather than encoding the entire decision tree as one monolithic disjunction, GlassBox decomposes `program.root` into disjoint leaf polyhedra $P_r = \{x \in [l, u] \mid G_r(x)\}$ and queries Z3 on each leaf $r$:

- **Regression (`PROVEN_REAL_GLOBAL`)**:
  $$\exists x \in P_r \text{ s.t. } \bigvee_{m=1}^{\text{out\_dim}} |NN_m(x) - \text{Leaf}_{r,m}(x)| > \varepsilon$$
- **Classification (`PROVEN_REAL_GLOBAL` when $\gamma = 0$, `PROVEN_REAL_MARGIN` when $\gamma > 0$)**:
  For predicted leaf class $c_r$ (or leaf score argmax), searches for $x \in P_r$ and rival class $j \ne c_r$ such that:
  $$NN_j(x) - NN_{c_r}(x) > \gamma$$

Every leaf where Z3 returns `unsat` is recorded in `RealArithmeticProof.leaf_proven_map[regime_id] = True`. If all leaves return `unsat` with zero exclusions, the program receives `PROVEN_REAL_GLOBAL` (or `PROVEN_REAL_MARGIN` when $\gamma > 0$).

---

## Independent Third-Party Proof Replay (`glassbox verify`)

A decompilation run with `--bundle` produces a self-contained `<model>.gbundle/` directory:

```text
drone_altitude_pid.gbundle/
├── manifest.json          # ArtifactManifest v2.0.0: SHA-256 of every file + tool_versions + config
├── model.onnx             # Source ONNX model
├── program.ir.json        # Canonical DecompiledProgram AST
├── certificate.json       # VerificationCertificate (RealArithmeticProof + NumericalDeploymentAssurance)
└── code/
    ├── drone_altitude_pid.c
    ├── drone_altitude_pid.h
    ├── drone_altitude_pid.rs
    └── drone_altitude_pid.py
```

Running `glassbox verify drone_altitude_pid.gbundle`:
1. Verifies every file's SHA-256 digest against `manifest.json` and `certificate.json` (`HASH_MISMATCH` on any tampering).
2. Reloads `model.onnx` and `program.ir.json` and **re-executes exact rational Z3 `verify_smt` from scratch**.
3. Re-verifies numerical ONNXRuntime vs. IR and emitted Python equivalence.
4. Returns `VERIFIED_REPRODUCED` (exit code `0`) only if all hashes match and the replayed proof reproduces the claimed `VerificationStatus` (`FALSIFIED_BY_REPLAY` if a forged certificate claims `PROVEN_REAL_GLOBAL` on an unverified or inaccurate program).
