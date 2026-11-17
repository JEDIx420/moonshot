# Project: Moonshot Developer and Research Tools Architecture & Execution Plan

## 1. Executive Summary & Architecture Evaluation
This project implements four core developer and research tools for the Moonshot project:
1. **R4: Micro-Autodiff Engine (`crates/moonshot-autodiff`)**: A pure Rust, zero-dependency reverse-mode automatic differentiation engine utilizing an explicit Wengert arena tape, verifying gradients against numerical finite differences for 8 fundamental operations (`add`, `sub`, `mul`, `matmul`, `gelu`, `softmax`, `layernorm`, `cross_entropy`).
2. **R5: Safetensors Exporter/Importer (`crates/moonshot-model/src/safetensors.rs`)**: A HuggingFace-compatible standard binary serialization layer supporting 77K and 1.09M models with 8-byte aligned JSON metadata and binary float buffer, verified via Python `safetensors.numpy` and Netron.
3. **R3: Moonshot Lab Interactive CLI REPL (`crates/moonshot-runner/src/bin/lab.rs`)**: A rich interactive CLI in Rust providing token-by-token generation stepping, ANSI syntax-highlighted output, real-time attention matrix inspection, and live sandbox register/memory state viewing.
4. **R2: Moonshot Cockpit Dashboard (`cockpit/server.py` + `cockpit/index.html`)**: A lightweight, zero-external-dependency local web dashboard (Python `http.server` + HTML/Chart.js) visualizing live retention heatmaps, learning curves, AST program lineage DAGs, and Kaggle run tracking from `experiments/run_ledger.jsonl` and CSV matrices.

---

## 2. Technical Evaluation, Edge Cases & Pitfall Analysis

### 2.1 Micro-Autodiff Engine (`crates/moonshot-autodiff`)
- **Architecture Choice**: Flat Wengert Arena (`Vec<TapeNode>`) with integer handles (`Var(usize)`).
  - *Tradeoff Analysis*: Dynamic graph structures (`Rc<RefCell<Node>>`) introduce pointer indirection, risk cyclic memory leaks, and require complex topological sorting algorithms. An arena provides topological ordering by construction (node indices strictly ascend: $i > \text{parents}(i)$), allowing linear $O(N)$ reverse iteration for adjoint evaluation.
- **Critical Edge Cases & Pitfalls**:
  1. *Unstable Nightly `float_erf`*: Rust `std::f32::erf` is an unstable nightly API (`float_erf`). Using it would fail standard stable compilation. *Solution*: Use the standard Hendrycks & Gimpel (2016) Tanh approximation $0.5x(1 + \tanh(\sqrt{2/\pi}(x + 0.044715x^3)))$, which is 100% stable in standard Rust and matches GPT-2/Transformers.
  2. *Fan-Out Gradient Accumulation*: Variables used in multiple downstream ops (e.g. residual connections or $y = x \cdot x$) must accumulate gradients (`+=`), not overwrite. *Solution*: Tape maintains an accumulator `grads: Vec<Option<Tensor>>` with `add_assign`.
  3. *Softmax/Cross-Entropy Numerical Stability*: Exponentiation overflows `f32` when logits $> 88$. *Solution*: Row-wise max subtraction ($x_i - \max(x)$) for softmax and stable LogSumExp formulation for cross-entropy.
  4. *LayerNorm Backward Gradient*: Involves interactions between normalized inputs $\hat{x}$, mean $\mu$, variance $\sigma^2$, and incoming $\bar{Y}$. *Solution*: Exact analytical adjoint formulation caching $\text{std\_inv}$ and intermediate projection sums.
  5. *Zero Dependencies*: `Cargo.toml` must have an empty `[dependencies]` section. Only standard library primitives and collections are permitted.

### 2.2 Safetensors Model Exporter/Importer (`crates/moonshot-model`)
- **Architecture Choice**: Direct binary parser and serializer in `src/safetensors.rs` utilizing existing `serde` / `serde_json` dependencies.
- **Critical Edge Cases & Pitfalls**:
  1. *8-Byte Memory Alignment*: HuggingFace `safe_open` and Netron expect the raw buffer offset $(8 + N)$ to be a multiple of 8. *Solution*: Pad the UTF-8 JSON header with trailing spaces (`0x20`) so that $N \pmod 8 == 0$.
  2. *Strict String-to-String Metadata*: Python's `safetensors` parser rejects nested objects or non-string values in `__metadata__`. *Solution*: Flatten all model config hyperparameters into string values (`BTreeMap<String, String>`).
  3. *LayerNorm Naming for Netron*: Netron expects `.weight` and `.bias` for normalization layers to render structured neural network graphs. *Solution*: Map internal `gamma` and `beta` to `.weight` and `.bias` on export; accept both canonical and legacy alias names on import.
  4. *Float Count Parity*: All 38 named tensors in both 77K (77,376 params) and 1.09M (1,095,744 params) models have even parameter counts, ensuring all tensor slice byte offsets remain 8-byte aligned.

### 2.3 Moonshot Lab Interactive CLI REPL (`crates/moonshot-runner`)
- **Architecture Choice**: Standalone binary `crates/moonshot-runner/src/bin/lab.rs` using standard library I/O and ANSI escape codes.
- **Critical Edge Cases & Pitfalls**:
  1. *Attention Weight Extraction Without Breaking Existing APIs*: `forward_multihead_attention` computes `(attn_out, attn_weights)` but `forward` discards it. *Solution*: Expose `forward_with_attention(&self, tokens: &[usize]) -> (Vec<f32>, Vec<Vec<f32>>)` on `TinyTransformer` so the REPL inspects full layer-by-layer, head-by-head attention without affecting existing callers.
  2. *Live Uncommitted Sandbox State*: Sandbox execution should reflect register changes as soon as instructions (`SET`, `ADD`, `COPY`) are generated, even before the model emits `COMMIT`. *Solution*: Pass accumulated tokens to `SandboxState::execute_tokens` at each step; display intermediate registers and error states.
  3. *Non-Interactive & Scripted Verification*: The REPL must not hang in automated test environments. *Solution*: Gracefully handle stdin EOF and parse scripted newline-separated commands (`echo "step\nregs\nquit" | cargo run --bin lab`).

### 2.4 Moonshot Cockpit Dashboard (`cockpit/`)
- **Architecture Choice**: Pure Python standard library `http.server.ThreadingHTTPServer` serving a single-page responsive interface with Chart.js.
- **Critical Edge Cases & Pitfalls**:
  1. *Zero External Python Dependencies*: Avoiding Flask, FastAPI, or Pandas ensures the cockpit runs out-of-the-box on any machine or remote server without `pip install`.
  2. *Streaming & Incomplete JSONL Writes*: Experiments running concurrently append to `run_ledger.jsonl`. *Solution*: Read line-by-line; skip trailing incomplete or corrupted lines with robust exception handling.
  3. *Missing Data Fallbacks*: If experiment directories or CSV matrices are absent, fall back gracefully to empty datasets and provide an automated `--mock` synthetic generator for local visual testing.

---

## 3. Feature Inventory

| # | Feature | Description | Milestone | Status | Source |
|---|---------|-------------|-----------|--------|--------|
| F1 | Workspace Member Integration | Add `crates/moonshot-autodiff` to workspace `members` in root `Cargo.toml` | M1 | DONE | R4 |
| F2 | Zero-Dependency Tape Engine | Pure Rust Wengert tape with `Var` handles, topological backward traversal | M1 | DONE | R4 |
| F3 | Linear & Elementwise Ops | Forward and backward adjoints for `add`, `sub`, `mul` with broadcasting | M1 | DONE | R4 |
| F4 | Matrix Multiplication | Forward and zero-allocation transposed backward for `matmul` | M1 | DONE | R4 |
| F5 | Tanh-GELU Activation | Numerically stable Tanh-approximation GELU and exact analytical adjoint | M1 | DONE | R4 |
| F6 | Stable Softmax & LogSumExp Cross-Entropy | Max-subtraction Softmax and LogSumExp Cross-Entropy with $(P - T)$ gradient | M1 | DONE | R4 |
| F7 | Layer Normalization | Row-wise mean/variance normalization and analytical scale/shift/input adjoints | M1 | DONE | R4 |
| F8 | Finite-Difference Gradcheck Suite | Automated central-difference gradient checker verifying all 8 ops within $10^{-3}$ tolerance | M1 | DONE | R4 |
| F9 | Safetensors Serialization Spec | 8-byte length prefix + space-padded JSON header + aligned float buffer | M2 | DONE | R5 |
| F10 | 77K & 1.09M Model Export | Full 38-tensor parameter extraction for both 77,376 and 1,095,744 parameter models | M2 | DONE | R5 |
| F11 | Robust Safetensors Importer | Safe validation of header bounds, dtype F32, offsets, and tensor reconstitutions | M2 | DONE | R5 |
| F12 | Netron & Python Compatibility | Netron visual hierarchy naming and verified `safetensors.numpy` loading | M2 | DONE | R5 |
| F13 | Transformer Attention Exposure | Non-breaking `forward_with_attention` exposing `[H x S x S]` per layer | M3 | DONE | R3 |
| F14 | Lab Interactive REPL Shell | `cargo run --bin lab` command loop (`step`, `run`, `reset`, `inspect`, `quit`) | M3 | DONE | R3 |
| F15 | ANSI Terminal Visualization | Syntax-highlighted tokens, box-bordered register/memory cards, attention heatmaps | M3 | DONE | R3 |
| F16 | Live Sandbox State Tracking | Real-time `SandboxState::execute_tokens` register & memory updates per step | M3 | DONE | R3 |
| F17 | Pure Python Cockpit Server | Standard library HTTP server with REST endpoints for ledger, retention, curves, lineage | M4 | DONE | R2 |
| F18 | Kaggle Run Tracking UI | Searchable, filterable table for `run_ledger.jsonl` with status badges & narratives | M4 | DONE | R2 |
| F19 | Live Retention Heatmaps | Dynamic 2D matrix rendering retention percentages across tasks and conditions | M4 | DONE | R2 |
| F20 | Learning Curves Visualization | Chart.js multi-line charts comparing learned vs baseline vs heuristic training runs | M4 | DONE | R2 |
| F21 | AST Program Lineage DAG | Visual tree/dag representation of evolved DSL update rules | M4 | DONE | R2 |
| F22 | E2E Automated Verification | Opaque-box test suite validating all 4 tools, CLI pipes, python imports, and netron files | M5 | DONE | AC |

---

## 4. Milestones & Task Allocation

| # | Name | Scope | Dependencies | Status | Key Outputs |
|---|------|-------|-------------|--------|-------------|
| M1 | Micro-Autodiff Engine | `crates/moonshot-autodiff/*`, root `Cargo.toml` | none | DONE | 22 unit tests passing, zero external dependencies, finite-diff verified |
| M2 | Safetensors Exporter/Importer | `crates/moonshot-model/src/safetensors.rs`, `src/lib.rs` | none | DONE | 18 tests passing, Python safetensors.numpy verified, Netron 38-tensor hierarchy |
| M3 | Moonshot Lab Interactive REPL | `crates/moonshot-runner/src/bin/lab.rs`, `crates/moonshot-runner/Cargo.toml` | none | DONE | 8 tests passing, ANSI terminal output, attention inspection, live sandbox VM |
| M4 | Moonshot Cockpit Dashboard | `cockpit/server.py`, `cockpit/index.html`, `cockpit/mock_data.py` | none | DONE | 14 tests passing, pure Python stdlib, live retention/curves/lineage/Kaggle runs |
| M5 | E2E Testing & Hardening | Master test suite `tests/*`, `TEST_INFRA.md`, `TEST_READY.md` | M1, M2, M3, M4 | DONE | 4/4 tiers passing 100% in `bash tests/run_all_e2e.sh`, adversarial checks passed |

---

## 5. Code Layout & Non-Conflicting Write Ownership

To guarantee zero file or workspace conflicts during concurrent execution, each worker was assigned **strictly disjoint write boundaries**:

```
/Users/vincyvincent/moonshot/
├── Cargo.toml                                      <-- Owned exclusively by Worker 1 (Autodiff)
├── crates/
│   ├── moonshot-autodiff/                          <-- Owned exclusively by Worker 1 (Autodiff)
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── tensor.rs
│   │       ├── tape.rs
│   │       ├── ops.rs
│   │       ├── error.rs
│   │       └── gradcheck.rs
│   ├── moonshot-model/                             <-- Owned exclusively by Worker 2 (Safetensors)
│   │   └── src/
│   │       ├── safetensors.rs
│   │       └── lib.rs (pub mod safetensors; & forward_with_attention)
│   └── moonshot-runner/                            <-- Owned exclusively by Worker 3 (Lab REPL)
│       ├── Cargo.toml ([[bin]] name = "lab")
│       └── src/
│           └── bin/
│               └── lab.rs
├── cockpit/                                        <-- Owned exclusively by Worker 4 (Cockpit)
│   ├── server.py
│   ├── index.html
│   ├── mock_data.py
│   └── test_cockpit.py
└── tests/                                          <-- Owned exclusively by E2E Testing Track
    ├── test_autodiff_e2e.py
    ├── test_safetensors_python.py
    ├── test_lab_repl_e2e.sh
    ├── test_cockpit_e2e.py
    └── run_all_e2e.sh
```

---

## 6. Interface Contracts

### 6.1 `moonshot-autodiff` Public API
```rust
pub struct Tensor { pub shape: Vec<usize>, pub data: Vec<f32> }
pub struct Var(pub usize);
pub struct Tape;
impl Tape {
    pub fn new() -> Self;
    pub fn var(&mut self, tensor: Tensor) -> Var;
    pub fn add(&mut self, a: Var, b: Var) -> Result<Var, AutodiffError>;
    pub fn sub(&mut self, a: Var, b: Var) -> Result<Var, AutodiffError>;
    pub fn mul(&mut self, a: Var, b: Var) -> Result<Var, AutodiffError>;
    pub fn matmul(&mut self, a: Var, b: Var) -> Result<Var, AutodiffError>;
    pub fn gelu(&mut self, a: Var) -> Result<Var, AutodiffError>;
    pub fn softmax(&mut self, a: Var, axis: usize) -> Result<Var, AutodiffError>;
    pub fn layernorm(&mut self, a: Var, gamma: Var, beta: Var, eps: f32) -> Result<Var, AutodiffError>;
    pub fn cross_entropy(&mut self, logits: Var, targets: &[usize]) -> Result<Var, AutodiffError>;
    pub fn backward(&self, loss: Var) -> Result<Gradients, AutodiffError>;
}
```

### 6.2 `moonshot-model` Safetensors API
```rust
impl TinyTransformer {
    pub fn to_safetensors(&self) -> Result<Vec<u8>, SafetensorsError>;
    pub fn from_safetensors(bytes: &[u8]) -> Result<Self, SafetensorsError>;
    pub fn forward_with_attention(&self, tokens: &[usize]) -> (Vec<f32>, Vec<Vec<f32>>);
}
```

### 6.3 Cockpit REST API Specification
- `GET /api/health` -> `{"status": "ok"}`
- `GET /api/ledger` -> `[ { "run_id": "...", "timestamp": "...", "status": "COMPLETE", ... } ]`
- `GET /api/retention` -> `[ { "condition": "...", "task": "...", "retention_ratio": 0.833, ... } ]`
- `GET /api/curves` -> `[ { "trial": 1, "learned_loss": 0.42, "raw_reward": 0.85, ... } ]`
- `GET /api/lineage` -> `[ { "condition": "...", "best_val_score": 0.92, "program": {...} } ]`

---

## 7. Parallel Multi-Agent Execution Plan & Verification Summary
All four tools were built and verified concurrently with strict file isolation:
1. **Worker 1 (Autodiff Engine)**: Completed `crates/moonshot-autodiff` with zero external dependencies and 22 passing tests.
2. **Worker 2 (Safetensors)**: Completed `crates/moonshot-model/src/safetensors.rs` with 18 passing tests and Python cross-loading.
3. **Worker 3 (Lab REPL)**: Completed `crates/moonshot-runner/src/bin/lab.rs` with 8 passing tests, ANSI formatting, and live sandbox inspection.
4. **Worker 4 (Cockpit Dashboard)**: Completed `cockpit/server.py` and `cockpit/index.html` with 14 passing tests and zero external pip dependencies.
5. **E2E Testing Track**: Published `TEST_INFRA.md` and `TEST_READY.md`; master test runner `bash tests/run_all_e2e.sh` passes 100% (4/4 tiers).
6. **Dual Review**: Reviewer 1 (Rust Tools) and Reviewer 2 (Cockpit & E2E) both issued **APPROVE** verdicts.
7. **Dual Challenger**: Challenger 1 (Numerical Stress & Fuzzing) and Challenger 2 (REPL Piping & Cockpit Concurrency) both issued **APPROVE** verdicts.
8. **Forensic Integrity Audit**: Forensic Auditor certified **CLEAN** with zero hardcoded facades, zero dummy shortcuts, and 100% genuine implementations.
