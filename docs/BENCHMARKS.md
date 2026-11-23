# Benchmarks and verification results

The four saved Studio runs contain one global real-arithmetic proof and three programs with known counterexamples. The snapshot was generated on **23 November 2026 at 09:37 UTC**, using GlassBox 0.2.0, Python 3.11.9, ONNX Runtime 1.32.0 and Z3 5.4.1. Each model link below opens the run that supplies its certificate, code and measured timings. [Machine-readable summary](../studio/demo-data/benchmarks_summary.json)

## What each run established

| Saved run | Task (inputs → outputs) | Parameters | Certificate status | Solver result | Regimes | AST nodes |
| --- | --- | ---: | --- | --- | ---: | ---: |
| [`drone_altitude_pid`](../studio/demo-data/runs/drone_altitude_pid.json) | Regression (3 → 1) | 189 | `PROVEN_REAL_GLOBAL` | UNSAT | 9 | 134 |
| [`cartpole_policy`](../studio/demo-data/runs/cartpole_policy.json) | Classification (4 → 2) | 242 | `COUNTEREXAMPLE_FOUND` | SAT | 3 | 107 |
| [`reactor_safety_guard`](../studio/demo-data/runs/reactor_safety_guard.json) | Classification (4 → 3) | 289 | `COUNTEREXAMPLE_FOUND` | SAT | 30 | 1,589 |
| [`nonlinear_actuator`](../studio/demo-data/runs/nonlinear_actuator.json) | Regression (3 → 2) | 746 | `COUNTEREXAMPLE_FOUND` | Skipped | 12 | 945 |

For the drone model, Z3 discharged all nine leaves with an absolute error tolerance of 0.05 across the declared domain. Its proof covers rational arithmetic on the extracted network and program. The floating-point deployments have separate sampled checks.

The CartPole and reactor runs each found a class disagreement satisfying their configured score margin (0.15 and 0.10, respectively). The nonlinear actuator used adversarial search and found an output error above 0.05.

| Saved run | Evaluation samples | Agreement | Largest observed absolute error | Mean absolute error | Witnesses across iterations |
| --- | ---: | ---: | ---: | ---: | ---: |
| `drone_altitude_pid` | 5,662 | 100.000% | 0.034540 | 0.002690 | 4 |
| `cartpole_policy` | 4,413 | 99.977% | 8.151106 | 1.195487 | 60 |
| `reactor_safety_guard` | 5,264 | 92.857% | 20.010329 | 1.942102 | 198 |
| `nonlinear_actuator` | 6,083 | 99.671% | 0.147855 | 0.008746 | 180 |

Agreement counts samples within tolerance for regression and matching predicted classes for classification. Absolute error uses model outputs, including classifier scores. Witnesses accumulate across synthesis iterations. All four logged drone witnesses were repaired in the saved program.

## CPU inference timings

Each percentile is reported in **microseconds per sample**. Batch-64 rows divide elapsed batch time by 64. The C99 FFI row times one call from Python to the compiled library. The native C99 row times repeated evaluations inside C, amortizing one Python call over the entire loop. Choose the row that matches how your application calls the model.

The single-call measurements use 100 warm-ups and 2,000 timed calls by default. Batch-64 measurements use 25 warm-ups and 250 timed batches. Native timing takes percentiles across 25 loops of 10,000 evaluations each. Throughput comes from elapsed wall time. The speedup column divides the batch-one ONNX Runtime median by each row’s per-sample median. [Benchmark harness](../src/glassbox/benchmark.py)

The maximum difference column uses 200 deterministic domain samples. ONNX Runtime and PyTorch use the loaded ONNX model as the reference. Emitted Python and C99 use the decompiled IR as the reference. The recorded C compiler was `ziglang cc` with `-O3`. The runs record software versions but omit the CPU model. The timing tables describe that recorded environment.

### `drone_altitude_pid`

| Runtime | p50 (µs) | p95 (µs) | p99 (µs) | Samples/s | Artifact (bytes) | p50 speedup | Max diff vs. reference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ONNX Runtime, batch 1 | 8.7 | 25.2 | 38.703 | 95,479 | 1,390 | 1.00× | 0 |
| ONNX Runtime, batch 64 | 0.16562 | 0.47219 | 0.58297 | 5,921,978 | 1,390 | 52.53× | 0 |
| PyTorch eager, batch 1 | 23.85 | 56.5 | 91.315 | 39,028 | 3,605 | 0.36× | 1.79e-07 |
| Emitted Python | 0.3 | 0.4 | 0.5 | 3,479,471 | 6,020 | 29.00× | 0 |
| C99, ctypes FFI | 0.5 | 0.9 | 1.2 | 1,838,573 | 224,256 | 17.40× | 2.22e-16 |
| C99, native loop | 0.00351 | 0.003882 | 0.004888 | 284,835,365 | 224,256 | 2478.63× | 2.22e-16 |

### `cartpole_policy`

| Runtime | p50 (µs) | p95 (µs) | p99 (µs) | Samples/s | Artifact (bytes) | p50 speedup | Max diff vs. reference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ONNX Runtime, batch 1 | 9 | 9.605 | 32.603 | 107,245 | 1,602 | 1.00× | 0 |
| ONNX Runtime, batch 64 | 0.11875 | 0.24766 | 0.51489 | 5,432,568 | 1,602 | 75.79× | 0 |
| PyTorch eager, batch 1 | 22.9 | 57.2 | 96.807 | 38,734 | 3,733 | 0.39× | 3.81e-06 |
| Emitted Python | 0.4 | 0.7 | 0.8 | 1,244,633 | 4,483 | 22.50× | 0 |
| C99, ctypes FFI | 0.7 | 0.8 | 0.9 | 2,005,415 | 225,280 | 12.86× | 7.11e-15 |
| C99, native loop | 0.00316 | 0.003392 | 0.0034076 | 312,773,677 | 225,280 | 2848.10× | 7.11e-15 |

### `reactor_safety_guard`

| Runtime | p50 (µs) | p95 (µs) | p99 (µs) | Samples/s | Artifact (bytes) | p50 speedup | Max diff vs. reference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ONNX Runtime, batch 1 | 6.3 | 9.7 | 24.703 | 130,809 | 1,792 | 1.00× | 0 |
| ONNX Runtime, batch 64 | 0.15781 | 0.17031 | 0.22908 | 5,771,798 | 1,792 | 39.92× | 0 |
| PyTorch eager, batch 1 | 16 | 26 | 59.011 | 41,837 | 3,989 | 0.39× | 1.24e-05 |
| Emitted Python | 0.7 | 0.9 | 1.2 | 1,591,723 | 31,706 | 9.00× | 0 |
| C99, ctypes FFI | 0.5 | 0.8 | 0.8 | 1,780,468 | 233,984 | 12.60× | 2.84e-14 |
| C99, native loop | 0.0065 | 0.00655 | 0.0065576 | 167,504,188 | 233,984 | 969.23× | 2.84e-14 |

### `nonlinear_actuator`

| Runtime | p50 (µs) | p95 (µs) | p99 (µs) | Samples/s | Artifact (bytes) | p50 speedup | Max diff vs. reference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ONNX Runtime, batch 1 | 8.7 | 9.6 | 10.012 | 115,502 | 3,620 | 1.00× | 0 |
| ONNX Runtime, batch 64 | 0.12969 | 0.22687 | 0.50414 | 7,732,831 | 3,620 | 67.08× | 0 |
| PyTorch eager, batch 1 | 16 | 25.205 | 53.701 | 47,810 | 5,781 | 0.54× | 4.77e-07 |
| Emitted Python | 0.5 | 0.8 | 0.9 | 2,110,372 | 14,146 | 17.40× | 0 |
| C99, ctypes FFI | 0.7 | 0.8 | 3.001 | 1,422,374 | 227,328 | 12.43× | 4.44e-16 |
| C99, native loop | 0.00414 | 0.0042 | 0.0045344 | 239,900,202 | 227,328 | 2101.45× | 4.44e-16 |

## Reproduce measurements locally

From the repository root, measure the committed drone model and IR on your hardware.

```sh
glassbox bench models/drone_altitude_pid.onnx bundles/drone_altitude_pid.gbundle/program.ir.json
```

To repeat the full model-zoo workflow, the generator loads the models, runs synthesis and verification, measures the runtimes and writes the Studio snapshots and the drone bundle. Running it updates tracked data files. [Generation script](../scripts/generate_demo_artifacts.py)

```sh
python scripts/generate_demo_artifacts.py
```
