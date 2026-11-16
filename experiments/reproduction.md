# Reproduction Guide: Moonshot Autonomous Neural Learner

**Repository**: `https://github.com/JEDIx420/moonshot`  
**License**: MIT / Apache 2.0  
**Authors**: Antigravity Research System & Moonshot Team  

---

## 1. Quick Start: Local Mathematical Integrity & Unit Verification

To verify mathematical correctness (multi-head causal self-attention gradients, analytical vs finite-difference tests, checkpoint bitwise determinism) on any workstation:

```bash
# Clone the repository
git clone https://github.com/JEDIx420/moonshot.git
cd moonshot

# Run all unit tests, property checks, and finite-difference gradient verifications
cargo test --release -- --nocapture
```

Expected output:
```
test tests::test_exact_finite_difference_gradients ... ok
test tests::test_checkpoint_bitwise_determinism ... ok
test tests::test_sandbox_state_execution ... ok
...
test result: ok. All tests passed.
```

---

## 2. Remote Kaggle Execution (Zero Local Mac Compute Policy)

In accordance with strict project constraints, all heavy neural training, meta-learning sweeps, and multi-generation evaluations are executed on Kaggle remote instances.

### 2.1 Prerequisites
- Kaggle API installed: `pip install kaggle`
- Kaggle credentials configured at `~/.kaggle/kaggle.json`

### 2.2 Execution via Kaggle CLI

To trigger an autonomous research wave on Kaggle:

```bash
# 1. Push kernel source to Kaggle
kaggle kernels push -p kaggle-cpu

# 2. Check running status
kaggle kernels status jayanandj/moonshot-cpu-runner

# 3. Retrieve execution logs and results once complete
kaggle kernels output jayanandj/moonshot-cpu-runner -p /tmp/moonshot_output
```

---

## 3. Direct Binary CLI Commands (Inside Kaggle Container)

The release binary `moonshot` provides direct CLI entrypoints for each experimental wave:

### Wave 0: Baseline & Gradient Correctness
```bash
moonshot smoke --output-dir /kaggle/working/results --git-commit $(git rev-parse HEAD)
moonshot baselines --output-dir /kaggle/working/results --seed 42 --target-acc 0.80
```

### Wave 1: Learned Proposal Policy Factorial Sweep
```bash
moonshot wave1-sweep --output-dir /kaggle/working/results --seed 42 --trials 25 --target-acc 0.80
```

### Wave 2: Credit Routing & Plasticity Gating Sweep
```bash
moonshot wave2-sweep --output-dir /kaggle/working/results --seed 42 --target-acc 0.80
```

### Wave 3: Compounding Multi-Generation Learning Sweep
```bash
moonshot wave3-sweep --output-dir /kaggle/working/results --seed 42 --target-acc 0.80
```

### Wave 4: Applied Multi-Step Tool Agent Sandbox Sweep
```bash
moonshot wave4-sweep --output-dir /kaggle/working/results --seed 42 --target-acc 0.80
```

---

## 4. Key Artifact Locations

- **Run Ledger**: `experiments/run_ledger.jsonl` (Chronological machine-readable ledger of every executed Kaggle run)
- **Wave 0**: `experiments/20261116-wave0-integrity/`
- **Wave 1**: `experiments/20261116-wave1-learned-editor/`
- **Wave 2**: `experiments/20261116-wave2-credit-routing/`
- **Wave 3**: `experiments/20261116-wave3-compounding/`
- **Wave 4**: `experiments/20261116-wave4-applied-sandbox/`
- **Master Report**: `experiments/FINAL_RESEARCH_REPORT.md`
