# Moonshot Developer & Research Tools: Test Infrastructure & 4-Tier E2E Methodology

## 1. Overview & Architectural Principles

The Moonshot testing infrastructure provides an **independent, opaque-box, multi-tier verification harness** covering all four developer and research tools specified in `ORIGINAL_REQUEST.md` and `PROJECT.md`.

The testing philosophy adheres to four foundational tenets:
1. **Opaque-Box Independence**: E2E test suites reside in `tests/` and evaluate compiled binaries, library interfaces, binary file artifacts, and HTTP network services without relying on internal private state or mocking facades.
2. **Authoritative Mathematical & Toolchain Oracles**: Every test derivation relies on an external authoritative oracle:
   - *Tier 1 (Autodiff)*: Central finite differences with step $\epsilon = 10^{-3}$ and double-precision reference math.
   - *Tier 2 (Safetensors)*: Official HuggingFace `safetensors.numpy` loader, standard IEEE-754 binary parsing, and Netron visual hierarchy schemas.
   - *Tier 3 (Lab REPL)*: Piped stdin execution validating ANSI VT100 control sequences, attention matrix tensor contracts, and `SandboxState` register/memory transitions.
   - *Tier 4 (Cockpit Dashboard)*: Python `http.client` / `urllib` asserting REST contract schemas, HTTP status codes, JSONL streaming tolerance, and dynamic port isolation.
3. **Adversarial & Boundary Stress Testing**: Tests proactively inject edge conditions: truncated binary streams, non-aligned headers, malformed JSONL, extreme logit values ($> 88.0$), fan-out gradient accumulation, and non-interactive EOF terminations.
4. **Zero-Dependency Reproducibility**: All test scripts utilize either the Python standard library or standard Rust toolchains, executing cleanly in standard macOS, Linux, and headless CI environments.

---

## 2. 4-Tier Test Architecture

```
                    ┌────────────────────────────────────────────────────────┐
                    │               Master E2E Test Runner                   │
                    │               (tests/run_all_e2e.sh)                   │
                    └──────────────────────────┬─────────────────────────────┘
                                               │
      ┌──────────────────┬─────────────────────┼─────────────────────┬──────────────────┐
      ▼                  ▼                     ▼                     ▼                  ▼
┌───────────────┐ ┌───────────────┐ ┌────────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│    Tier 1     │ │    Tier 2     │ │       Tier 2       │ │      Tier 3      │ │      Tier 4      │
│   Autodiff    │ │  Safetensors  │ │    Safetensors     │ │     Lab REPL     │ │     Cockpit      │
│  Gradcheck &  │ │ Pure Binary   │ │ Official Python    │ │ Interactive CLI  │ │ REST API & Web   │
│ Composite E2E │ │ Specification │ │ safetensors Oracle │ │ Piped Automation │ │ Single-Page App  │
│  (Py + Rust)  │ │   (Python)    │ │   (uv safetensors) │ │     (Bash)       │ │     (Python)     │
└───────────────┘ └───────────────┘ └────────────────────┘ └──────────────────┘ └──────────────────┘
```

### 2.1 Tier 1: Micro-Autodiff Engine Numerical Verification
- **Target**: `crates/moonshot-autodiff` (Feature items F1, F2, F3, F4, F5, F6, F7, F8).
- **Core Verifications**:
  1. **Zero External Dependencies**: Validates that `crates/moonshot-autodiff/Cargo.toml` contains an empty `[dependencies]` section.
  2. **8 Elementary Operations**: Compares analytical reverse-mode tape gradients against central finite differences for:
     - `add(lhs, rhs)` with 2D row broadcasting
     - `sub(lhs, rhs)` with 2D row broadcasting
     - `mul(lhs, rhs)` elementwise Hadamard multiplication & self-multiplication ($x \cdot x$)
     - `matmul(lhs, rhs)` with forward inner product and zero-allocation transposed backward
     - `gelu(x)` using stable Tanh approximation ($0.5x(1 + \tanh(\sqrt{2/\pi}(x + 0.044715x^3)))$)
     - `softmax(x, axis)` with row-wise max subtraction numerical stabilization
     - `layernorm(x, gamma, beta, eps)` with exact scale, bias, and normalized input adjoints
     - `cross_entropy(logits, targets)` with LogSumExp stabilization and $(P - T)$ adjoint
  3. **Composite Graph Gradient Propagation**: Evaluates a complete 2-layer MLP (`Linear -> GELU -> LayerNorm -> Linear -> CrossEntropy`) verifying end-to-end chain rule accumulation.
  4. **Tolerance Criteria**:
     - Central difference step: $\epsilon = 10^{-3}$
     - Relative error metric: $\frac{|g_{\text{auto}} - g_{\text{num}}|}{\max(|g_{\text{auto}}|, |g_{\text{num}}|, 10^{-4})}$
     - Tolerance threshold: $\text{rel\_err} < 10^{-3}$ or $\text{abs\_diff} < 10^{-4}$.

### 2.2 Tier 2: Safetensors Model Exporter/Importer Verification
- **Target**: `crates/moonshot-model/src/safetensors.rs` (Feature items F9, F10, F11, F12).
- **Core Verifications**:
  1. **Binary Header Specification**:
     - 8-byte little-endian unsigned integer prefix $N = \text{len(header)}$.
     - Valid UTF-8 JSON header starting with `{` and padded with trailing ASCII spaces (`0x20`) such that $N \equiv 0 \pmod 8$.
     - Buffer offset $(8 + N) \equiv 0 \pmod 8$ for zero-copy memory-mapping alignment.
  2. **Model Parameter Parity**:
     - 77K Model: Exactly 38 named tensors, 77,376 float parameters ($309,504$ bytes).
     - 1.09M Model: Exactly 38 named tensors, 1,095,744 float parameters ($4,382,976$ bytes).
  3. **Netron Visual Hierarchy**:
     - Strict dot-separated naming: `token_emb.weight`, `pos_emb.weight`, `blocks.{l}.attn.w_q.weight`, `blocks.{l}.ffn.w_1.weight`, `blocks.{l}.attn.ln.weight`, `ln_f.weight`, `head.weight`.
  4. **Official Python Oracle Compatibility**:
     - Verified loading using Python's official `safetensors.numpy.load_file`.
     - Validates tensor shape preservation, F32 dtype, and float value bit-identity.
  5. **Adversarial Error Handling**:
     - Rejection of truncated files ($< 8$ bytes).
     - Rejection of corrupted header size prefixes.
     - Rejection of out-of-bounds tensor offsets.
     - Rejection of unsupported data types (non-F32).

### 2.3 Tier 3: Moonshot Lab REPL Verification
- **Target**: `crates/moonshot-runner/src/bin/lab.rs` (Feature items F13, F14, F15, F16).
- **Core Verifications**:
  1. **Scripted Piped Execution**: Executes `cargo run --bin lab` via piped stdin, proving non-interactive automation capability and graceful EOF termination without hanging.
  2. **REPL Command Set**:
     - `help`: Renders command summary and syntax options.
     - `step`: Generates a single token, prints the colored token, updates context.
     - `run`: Runs autoregressive prediction until EOS or sequence limit.
     - `attn`: Displays multi-head attention weights per layer.
     - `regs`: Displays current sandbox registers `[R0, R1, R2, R3]`.
     - `mem`: Displays current sandbox memory `[M0, M1, M2, M3]`.
     - `reset`: Clears generation context back to initial prompt.
     - `quit` / `exit`: Clean process termination with exit code 0.
  3. **ANSI Syntax Formatting**:
     - Validates presence of ANSI escape sequences for token categories (BOS green, EOS red, CMDs magenta, REGs cyan).
     - Validates box-drawing characters (`┌`, `─`, `┐`, `│`, `└`, `┘`) for register and memory cards.
  4. **Live Incremental VM State Tracking**:
     - Demonstrates that register modifications (`SET R0 5`, `ADD R0 R1 R2`) update live in the REPL even before a `COMMIT` token is emitted.

### 2.4 Tier 4: Moonshot Cockpit Dashboard Verification
- **Target**: `cockpit/server.py` + `cockpit/index.html` (Feature items F17, F18, F19, F20, F21).
- **Core Verifications**:
  1. **Standard Library Server Architecture**: Pure Python `http.server.ThreadingHTTPServer` with zero external pip dependencies.
  2. **REST API Contract Validation**:
     - `GET /api/health` -> HTTP 200, `{"status": "ok"}`
     - `GET /api/ledger` -> HTTP 200, JSON list of run records (`run_id`, `timestamp`, `kernel`, `status`, `outcome`, `finding`).
     - `GET /api/retention` -> HTTP 200, JSON list of retention matrix records (`condition`, `task`, `exact_match`, `retention_ratio`).
     - `GET /api/curves` -> HTTP 200, JSON list of learning curve records (`trial`, `task_family`, `loss`, `acc`).
     - `GET /api/lineage` -> HTTP 200, JSON structure with AST program evolution nodes and lineage DAG edges.
     - `GET /` -> HTTP 200, HTML content containing dashboard DOM mounts and Chart.js script elements.
  3. **Fault-Tolerant Streaming JSONL Reader**:
     - Simulates incomplete/partial trailing writes to `run_ledger.jsonl`; server skips corrupted lines without 500 Internal Server Errors.
  4. **Mock vs Real Modes**:
     - Validates `--mock` flag operation when `experiments/` directory is missing or empty.
     - Validates live operation against actual experiment files in `experiments/`.
  5. **Dynamic Port Isolation & Clean Lifecycle**:
     - Starts server on an ephemeral port (e.g. dynamic free port), runs full test suite, and cleanly terminates the server process.

---

## 3. Requirements Traceability Matrix

| Requirement | Feature | Description | E2E Test Suite | Oracle / Standard |
|---|---|---|---|---|
| **R4** | F1, F2 | Zero-dependency Tape Engine | `tests/test_autodiff_e2e.py` | Empty `[dependencies]`, Reverse-mode adjoint traversal |
| **R4** | F3 | Add, Sub, Mul adjoints | `tests/test_autodiff_e2e.py` | Central finite difference ($\epsilon = 10^{-3}$) |
| **R4** | F4 | MatMul forward & backward | `tests/test_autodiff_e2e.py` | Analytical transposed matrix gradient vs numerical |
| **R4** | F5 | Tanh-GELU & analytical adjoint | `tests/test_autodiff_e2e.py` | Hendrycks & Gimpel (2016) Tanh approximation |
| **R4** | F6 | Softmax & Cross-Entropy | `tests/test_autodiff_e2e.py` | Max-subtraction Softmax & LogSumExp Cross-Entropy $(P-T)$ |
| **R4** | F7 | Layer Normalization | `tests/test_autodiff_e2e.py` | Exact row-wise mean/variance adjoints |
| **R4** | F8 | Finite-difference suite | `tests/test_autodiff_e2e.py` | Central difference relative error $< 10^{-3}$ |
| **R5** | F9 | Safetensors 8-byte alignment | `tests/test_safetensors_python.py` | Little-endian $N$, space-padded JSON, $(8+N)\equiv 0 \pmod 8$ |
| **R5** | F10 | 77K & 1.09M Model export | `tests/test_safetensors_python.py` | 38 tensors, 77,376 & 1,095,744 params |
| **R5** | F11 | Robust Safetensors importer | `tests/test_safetensors_python.py` | Adversarial bounds, truncation, dtype error checks |
| **R5** | F12 | Python & Netron compatibility | `tests/test_safetensors_python.py` | `safetensors.numpy` loader & Netron hierarchy keys |
| **R3** | F13 | Multi-head attention exposure | `tests/test_lab_repl_e2e.sh` | Shape $[H, S, S]$ attention matrix inspection |
| **R3** | F14 | Lab REPL CLI shell | `tests/test_lab_repl_e2e.sh` | Interactive command loop (`step`, `run`, `reset`, `quit`) |
| **R3** | F15 | ANSI Terminal Visualization | `tests/test_lab_repl_e2e.sh` | ANSI colors, Unicode box-drawing, formatted tables |
| **R3** | F16 | Live Sandbox State Tracking | `tests/test_lab_repl_e2e.sh` | `SandboxState` register & memory updates |
| **R2** | F17 | Pure Python Cockpit Server | `tests/test_cockpit_e2e.py` | `http.server.ThreadingHTTPServer` REST endpoints |
| **R2** | F18 | Kaggle Run Tracking UI | `tests/test_cockpit_e2e.py` | `/api/ledger` schema & status pill verification |
| **R2** | F19 | Live Retention Heatmaps | `tests/test_cockpit_e2e.py` | `/api/retention` matrix parsing & values |
| **R2** | F20 | Learning Curves Visualization | `tests/test_cockpit_e2e.py` | `/api/curves` loss & reward trajectories |
| **R2** | F21 | AST Program Lineage DAG | `tests/test_cockpit_e2e.py` | `/api/lineage` program tree & validation scores |
| **AC** | F22 | E2E Automated Verification | `tests/run_all_e2e.sh` | Master runner orchestrating Tiers 1-4 |

---

## 4. Test Execution Instructions

### 4.1 Master E2E Runner (All Tiers)
Execute the unified test runner:
```bash
bash tests/run_all_e2e.sh
```

### 4.2 Individual Test Suites
```bash
# Tier 1: Micro-Autodiff Numerical Gradient & Dependency Verification
python3 tests/test_autodiff_e2e.py

# Tier 2: Safetensors Binary Alignment, Netron Hierarchy & Python Loader Verification
python3 tests/test_safetensors_python.py

# Tier 3: Moonshot Lab REPL Interactive Stepping & State Verification
bash tests/test_lab_repl_e2e.sh

# Tier 4: Moonshot Cockpit REST API & Single-Page Dashboard Verification
python3 tests/test_cockpit_e2e.py
```
