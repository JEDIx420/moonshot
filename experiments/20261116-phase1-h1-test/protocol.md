# Moonshot Preregistration: Phase 1 & 2 Hypothesis H1 Test
Date: 2026-11-16
Run ID: 20261116-phase1-h1-test
Author: Antigravity Autonomous Agent

## 1. Hypothesis H1
A tiny neural transformer whose learning rule state $\phi$ (layer learning rate multipliers, mistake-prioritized sampling, and warmup-cosine schedule) evolves through feedback-conditioned self-editing will require measurably fewer examples or inner steps to reach target accuracy on held-out task families than an identical learner using a fixed AdamW update policy at matched compute.

## 2. Experimental Conditions & Matched Baselines
1. **Frozen Baseline**: Pre-initialized model with zero adaptation steps. Evaluates lower-bound accuracy.
2. **Fixed Learner Baseline**: Standard AdamW optimizer with fixed learning rate (0.01) and uniform sampling.
3. **Random Edit Policy**: Matched search attempts with randomly mutated learning rules within safety bounds.
4. **Evolvable Self-Editing Learner**: $\phi$ proposed by editor $E$ conditioned on task feedback, sandboxed by typechecker, promoted only when adaptation efficiency beats the incumbent.

## 3. Tasks & Evaluation Protocol
- **Meta-Train Families**: FSM Permutation Graphs, Symbolic Primitives (Shift/Duplicate), Tool Workflow Basic.
- **Unopened Held-Out Meta-Test Families**:
  1. `FsmHierarchicalTree`: Multi-level binary tree state transitions with reset triggers (disjoint topology).
  2. `SymbolicCompositional`: Mirror-Duplicate compositional grammar.
  3. `ToolWorkflowChained`: Multi-step register arithmetic dependency chains.
- **Evaluation Seeds**: [101, 202, 303].
- **Primary Metric**: Samples to target accuracy (65% exact sequence match).
- **Secondary Metrics**: Final validation accuracy, retention on prior tasks, GPU execution seconds.

## 4. Quota Controls
- Starting GPU Quota: ~29.80 hours.
- Expected spend: ~0.08 GPU hours (~5 minutes).
- Safety reserve maintained: > 80% remaining.
