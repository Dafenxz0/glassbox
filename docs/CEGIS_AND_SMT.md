# CEGIS Synthesis & SMT Verification

GlassBox separates **program synthesis** from **formal verification** using a Counterexample-Guided Inductive Synthesis (CEGIS) loop, and strictly separates **formal SMT proofs (Tier A)** from **bounded empirical falsification (Tier B)**.

## Verification Tiers at a Glance

| Tier | Applicable Networks | Method | Max Guaranteed Claim |
| :--- | :--- | :--- | :--- |
| **Tier A (`PROVEN_SMT`)** | Piecewise-linear networks (e.g., `ReLU` MLPs) | Exact Z3 `QF_LRA` SMT encoding | Mathematical guarantee across the entire declared `DomainSpec` hyperrectangle (`UNSAT` = zero counterexamples within `epsilon` / class agreement) |
| **Tier B (`BOUNDED_EMPIRICAL`)** | Smooth / non-linear networks (`Tanh`, `Sigmoid`, `GELU`, `SiLU`) | Quasi-random sampling + adversarial gradient search | No counterexample found within the configured search budget; **not** a universal proof |
| **Unverified / Failed (`COUNTEREXAMPLE_FOUND` / `UNVERIFIED`)** | Any network | Residual violation remains or verifier did not run | Explicit counterexample input `x*` and discrepancy recorded |

GlassBox never promotes a Tier B result to `PROVEN_SMT`. Every `VerificationCertificate` records the exact tier, tolerance (`epsilon`), domain bounds, and iteration trace.

---

## Tier A: Exact `QF_LRA` Z3 Encoding (ReLU MLPs)

A feed-forward ReLU neural network on a bounded real domain $x \in [l, u] \subset \mathbb{R}^d$ is piecewise affine, meaning both the network and the synthesized `ProgramIR` live inside **Quantifier-Free Linear Real Arithmetic (`QF_LRA`)**.

### 1. Encoding the Neural Network
For each input dimension $i \in \{1, \dots, d\}$:
$$l_i \le x_i \le u_i$$

For each hidden neuron $j$ in layer $k$ with pre-activation $z_{k,j} = w_{k,j}^\top h_{k-1} + b_{k,j}$ and output $h_{k,j} = \max(0, z_{k,j})$:
$$h_{k,j} \ge 0, \quad h_{k,j} \ge z_{k,j}$$
$$(h_{k,j} = z_{k,j} \land z_{k,j} \ge 0) \lor (h_{k,j} = 0 \land z_{k,j} \le 0)$$

When interval arithmetic (IBP) establishes tight pre-activation bounds $[L_{k,j}, U_{k,j}]$:
- If $L_{k,j} \ge 0$, the neuron is **stably active** ($h_{k,j} = z_{k,j}$, no branching).
- If $U_{k,j} \le 0$, the neuron is **stably inactive** ($h_{k,j} = 0$, pruned).
- Otherwise, Z3 branches only on the uncertain neurons.

### 2. Encoding the Candidate Program
The synthesized `ProgramIR` tree consists of linear half-space guards ($a^\top x + c \ge 0$) and affine leaf expressions ($W_r x + b_r$). Each path condition and leaf expression is encoded directly as `QF_LRA` implications over the same symbolic input vector $x$.

### 3. Equivalence Negation Query
- **Regression (`TaskType.REGRESSION`)**: Z3 searches for any $x \in [l, u]$ such that:
  $$\bigvee_{m=1}^{\text{out\_dim}} |NN_m(x) - P_m(x)| > \varepsilon$$
- **Classification (`TaskType.CLASSIFICATION`)**: Z3 searches for any $x \in [l, u]$ where the predicted argmax class of $NN(x)$ differs from $P(x)$ (or where output margins diverge beyond `epsilon` when margin equivalence is requested).

If Z3 returns **`sat`**, the model extracted from Z3 gives a concrete counterexample $x^*$, along with exact `nn_output` and `program_output`. The CEGIS synthesizer ingests $x^*$ to refine the local regime partition and repeats. If Z3 returns **`unsat`**, no counterexample exists anywhere in $[l, u]$, and the certificate is sealed as `PROVEN_SMT`.

---

## Tier B: Adversarial Falsification (Smooth Networks)

Smooth activations such as $\tanh(z)$, $\sigma(z)$, and $\text{GELU}(z)$ introduce transcendental functions outside decidable `QF_LRA`. Rather than silently approximating them inside Z3 and claiming a proof, GlassBox routes smooth networks to **Tier B**:

1. **Quasi-random exploration**: Evaluates low-discrepancy samples across `[lower_bounds, upper_bounds]` to expose global regions of high discrepancy.
2. **Adversarial ascent**: Starts from the highest-error seeds and performs projected gradient / local perturbation steps inside `[lower_bounds, upper_bounds]` to maximize $\|NN(x) - P(x)\|_\infty$ (or induce a class flip).
3. **CEGIS feedback**: Any discovered violation above `epsilon` is returned as a `Counterexample` (`source="adversarial"` or `"sampling"`) to refine the synthesized program.
4. **Honest status**: When the search budget finishes with zero violations above `epsilon`, GlassBox emits `BOUNDED_EMPIRICAL` and records the sample count and worst observed error in the certificate.
