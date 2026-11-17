# Original User Request

## Initial Request — 2026-11-17T05:21:45Z

Cross-examine architectures, evaluate edge cases and pitfalls, and use the teamwork multi-agent capability effectively to plan, build, and verify four developer and research tools for the Moonshot project without conflicts.

Working directory: /Users/vincyvincent/moonshot
Integrity mode: development

## Requirements

### R1. Architecture Evaluation and Multi-Agent Execution Plan
Seriously cross-examine and evaluate technical approaches for all four tools, identify edge cases and pitfalls, design clean modular architectures, and formulate a parallel multi-agent execution plan ensuring workers build and verify simultaneously without file or dependency conflicts.

### R2. Moonshot Cockpit Dashboard
A local web dashboard (Python/HTML/Chart.js) reading `experiments/run_ledger.jsonl` and CSVs to display live retention heatmaps, learning curves, AST program lineage, and Kaggle run tracking.

### R3. Moonshot Lab Interactive CLI REPL
An interactive CLI REPL in Rust (`crates/moonshot-runner/src/bin/lab.rs`) to step token-by-token with ANSI terminal output, attention inspection, and sandbox register/memory state viewing.

### R4. Micro-Autodiff Engine
A zero-dependency reverse-mode tape in pure Rust (`crates/moonshot-autodiff`) with finite-difference verified gradients for add, sub, mul, matmul, gelu, softmax, layernorm, and cross_entropy.

### R5. Safetensors Model Exporter/Importer
HuggingFace-compatible standard binary serialization (`crates/moonshot-model/src/safetensors.rs`) for 77K and 1.09M models, compatible with Netron and Python.

## Verification Resources
- Use existing project files and test suites in `/Users/vincyvincent/moonshot`.
- Generate minimal synthetic fixtures or mock ledger/checkpoint files where missing to enable independent, automated verification.

## Acceptance Criteria

### Architectural Plan
- [ ] A documented architecture evaluation and parallel multi-agent execution plan detailing edge cases, component interfaces, and non-conflicting task allocations.

### Moonshot Cockpit
- [ ] Running the dashboard serves a local web interface rendering live retention heatmaps, learning curves, AST lineage, and Kaggle run tracking from `experiments/run_ledger.jsonl` and CSV data without frontend or backend errors.

### Moonshot Lab
- [ ] Running `cargo run --bin lab` launches the interactive REPL.
- [ ] Stepping token-by-token displays formatted ANSI terminal output.
- [ ] REPL inspection views display attention weights and sandbox register/memory states.

### Micro-Autodiff Engine
- [ ] `cargo test -p moonshot-autodiff` passes with zero external dependencies in `Cargo.toml`.
- [ ] Gradient calculations for add, sub, mul, matmul, gelu, softmax, layernorm, and cross_entropy match numerical finite-difference checks within a strict numerical tolerance.

### Safetensors Exporter/Importer
- [ ] `cargo test -p moonshot-model` passes tests for Safetensors serialization and deserialization.
- [ ] Exported weight files for 77K and 1.09M models can be read by standard Python Safetensors loaders and inspected in Netron.
- [ ] Round-trip export and import in Rust recovers identical tensor shapes, dtypes, and values.

Working directory: /Users/vincyvincent/moonshot
State folder: /Users/vincyvincent/moonshot/.agents/teamwork/orchestrator/
Original request: /Users/vincyvincent/moonshot/.agents/teamwork/ORIGINAL_REQUEST.md
