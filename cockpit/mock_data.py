"""Synthetic Mock Data for Moonshot Cockpit Dashboard.

Provides complete, realistic mock data conforming to Moonshot experiment schemas:
- experiments/run_ledger.jsonl
- retention_matrix*.csv
- learning_curves.csv / meta_curve.csv
- program_lineage.jsonl / update_rule_schema.json
"""

from typing import Any, Dict, List


def get_mock_ledger() -> List[Dict[str, Any]]:
    """Return synthetic run ledger entries matching run_ledger.jsonl schema."""
    return [
        {
            "run_id": "run_01_phase0_smoke_gpu_mock",
            "timestamp": "2026-11-16T12:43:08Z",
            "kernel": "jayanandj/moonshot-research-runner",
            "version": 1,
            "gpu_minutes": 3.4,
            "status": "COMPLETE",
            "outcome": "PASS_INFRASTRUCTURE",
            "finding": "Kaggle 2xT4 GPU verified. Pure-Rust toolchain and release compilation verified.",
        },
        {
            "run_id": "run_04_h1_test_cpu_calibrated_mock",
            "timestamp": "2026-11-16T13:25:24Z",
            "kernel": "jayanandj/moonshot-cpu-runner",
            "version": 4,
            "cpu_minutes": 5.8,
            "status": "COMPLETE",
            "outcome": "H1_SIGNAL_OBSERVED",
            "finding": "Evolved rule outperformed fixed baseline across all 3 held-out task families on 3 independent seeds. Sample efficiency improved 16.7%.",
        },
        {
            "run_id": "run_11_wave5b_forgetting_microscope_mock",
            "timestamp": "2026-11-16T20:57:42Z",
            "kernel": "jayanandj/moonshot-research-runner",
            "version": 12,
            "gpu_minutes": 6.8,
            "status": "COMPLETE",
            "outcome": "PASS_GATE_C1_C2",
            "finding": "Forgetting microscope identified catastrophic forgetting in standard SGD (retention dropped to 0.00).",
        },
        {
            "run_id": "run_13_wave5d_modular_adapters_mock",
            "timestamp": "2026-11-16T21:39:10Z",
            "kernel": "jayanandj/moonshot-research-runner",
            "version": 14,
            "gpu_minutes": 12.4,
            "status": "COMPLETE",
            "outcome": "WAVE5D_GATE_S1_PASSED",
            "finding": "Rank-4 modular adapters isolated interference; retention increased from 0.00 to 0.833 on sequential tool workflows.",
        },
        {
            "run_id": "run_14_wave5e_update_rule_dsl_mock",
            "timestamp": "2026-11-17T00:23:15Z",
            "kernel": "jayanandj/moonshot-research-runner",
            "version": 15,
            "gpu_minutes": 22.1,
            "status": "COMPLETE",
            "outcome": "WAVE5E_GATE_PASSED",
            "finding": "Evolved 3-op AST programs (ClipGradNorm -> AttenuateGroup -> WarmupCosineSchedule) discovered non-trivial regularization.",
        },
        {
            "run_id": "run_18_wave6b_1m_modular_subspace_mock",
            "timestamp": "2026-11-17T04:50:06Z",
            "kernel": "jayanandj/moonshot-cpu-runner",
            "version": 19,
            "cpu_minutes": 88.5,
            "status": "COMPLETE",
            "outcome": "WAVE6B_GATE_S1_PASSED",
            "finding": "1M parameter model verified on modular subspace adapters: 83.33% retention vs 0.00% for full fine-tuning.",
        },
    ]


def get_mock_retention() -> Dict[str, Any]:
    """Return synthetic retention matrix and raw rows."""
    conditions = [
        "1M_Rank4_Modular_Adapters",
        "1M_AGEM_Projection",
        "77K_Shared_Baseline",
        "1M_Full_FineTuning",
    ]
    tasks = ["ToolWorkflowBasic", "FsmHierarchicalTree", "ToolWorkflowSandbox"]

    # Matrix cell values: condition -> task -> {retention_ratio, exact_match, token_acc}
    matrix: Dict[str, Dict[str, Dict[str, float]]] = {
        "1M_Rank4_Modular_Adapters": {
            "ToolWorkflowBasic": {"retention_ratio": 0.8333, "exact_match": 0.850, "token_acc": 0.945},
            "FsmHierarchicalTree": {"retention_ratio": 0.8000, "exact_match": 0.720, "token_acc": 0.890},
            "ToolWorkflowSandbox": {"retention_ratio": 0.8667, "exact_match": 0.780, "token_acc": 0.920},
        },
        "1M_AGEM_Projection": {
            "ToolWorkflowBasic": {"retention_ratio": 0.4500, "exact_match": 0.520, "token_acc": 0.780},
            "FsmHierarchicalTree": {"retention_ratio": 0.3800, "exact_match": 0.410, "token_acc": 0.710},
            "ToolWorkflowSandbox": {"retention_ratio": 0.4200, "exact_match": 0.460, "token_acc": 0.750},
        },
        "77K_Shared_Baseline": {
            "ToolWorkflowBasic": {"retention_ratio": 0.2000, "exact_match": 0.300, "token_acc": 0.650},
            "FsmHierarchicalTree": {"retention_ratio": 0.1500, "exact_match": 0.220, "token_acc": 0.580},
            "ToolWorkflowSandbox": {"retention_ratio": 0.1800, "exact_match": 0.250, "token_acc": 0.610},
        },
        "1M_Full_FineTuning": {
            "ToolWorkflowBasic": {"retention_ratio": 0.0000, "exact_match": 0.020, "token_acc": 0.320},
            "FsmHierarchicalTree": {"retention_ratio": 0.0000, "exact_match": 0.010, "token_acc": 0.280},
            "ToolWorkflowSandbox": {"retention_ratio": 0.0000, "exact_match": 0.000, "token_acc": 0.250},
        },
    }

    rows = []
    for cond, task_map in matrix.items():
        for task, metrics in task_map.items():
            rows.append({
                "condition": cond,
                "eval_task": task,
                "retention_ratio": metrics["retention_ratio"],
                "exact_match": metrics["exact_match"],
                "token_acc": metrics["token_acc"],
                "seed": 42,
                "stage": 2,
            })

    return {
        "dataset_name": "synthetic_mock_retention_1m",
        "conditions": conditions,
        "tasks": tasks,
        "matrix": matrix,
        "rows": rows,
    }


def get_mock_curves() -> Dict[str, Any]:
    """Return synthetic learning curve series."""
    meta_trials = []
    baseline = 0.05
    for t in range(25):
        loss = round(max(0.12, 1.8 * (0.88 ** t) + 0.03 * ((t % 3) - 1)), 4)
        reward = round(min(0.98, 0.2 + 0.75 * (1.0 - (0.85 ** t)) + 0.02 * ((t % 4) - 2)), 4)
        baseline = round(baseline * 0.95 + reward * 0.05, 4)
        shuffled = round(max(0.05, 0.35 + 0.1 * ((t % 5) - 2) * 0.1), 4)
        meta_trials.append({
            "trial": t,
            "family": "tool_workflow_sandbox_test",
            "raw_reward": reward,
            "running_baseline": baseline,
            "learned_loss": loss,
            "shuffled_reward": shuffled,
        })

    # Wave1 condition comparison curves
    wave1_series = {
        "learned": [{"trial": i, "acc": round(min(1.0, 0.1 * i + 0.05), 3)} for i in range(10)],
        "heuristic": [{"trial": i, "acc": round(min(0.7, 0.07 * i + 0.05), 3)} for i in range(10)],
        "shuffled": [{"trial": i, "acc": round(min(0.35, 0.03 * i + 0.02), 3)} for i in range(10)],
        "random": [{"trial": i, "acc": round(min(0.2, 0.02 * i + 0.01), 3)} for i in range(10)],
        "frozen": [{"trial": i, "acc": 0.014} for i in range(10)],
    }

    return {
        "meta_curve": meta_trials,
        "wave1_curves": wave1_series,
    }


def get_mock_lineage() -> Dict[str, Any]:
    """Return synthetic AST program lineage graph and program definitions."""
    operators = [
        {"name": "ClipGradNorm", "args": {"max_norm": "float [0.1, 10.0]"}},
        {"name": "AttenuateGroup", "args": {"group": "int [0, 5]", "factor": "float [0.0, 1.0]"}},
        {"name": "ProtectGroup", "args": {"group": "int [0, 5]"}},
        {"name": "ReplayAnchorBatch", "args": {"ratio": "float [0.0, 0.5]"}},
        {"name": "WarmupCosineSchedule", "args": {"inner_steps": "int [20, 90]"}},
        {"name": "MomentumDecouple", "args": {"weight_decay": "float [0.0001, 0.1]"}},
        {"name": "IdentityNoOp", "args": {}},
    ]

    best_programs = [
        {
            "id": "prog_seed707",
            "condition": "Learned_DSL_Programs",
            "family": "tool_workflow_sandbox_test",
            "seed": 707,
            "best_val_score": 0.9879,
            "ops": [
                {"name": "ClipGradNorm", "params": {"max_norm": 1.0}},
                {"name": "AttenuateGroup", "params": {"group": 4, "factor": 0.3}},
                {"name": "WarmupCosineSchedule", "params": {"inner_steps": 55}},
            ],
            "description": "ClipGradNorm(1.0) -> AttenuateGroup(4, 0.3) -> WarmupCosineSchedule(55)",
        },
        {
            "id": "prog_seed808",
            "condition": "Learned_DSL_Programs",
            "family": "tool_workflow_sandbox_test",
            "seed": 808,
            "best_val_score": 1.1995,
            "ops": [
                {"name": "ClipGradNorm", "params": {"max_norm": 1.0}},
                {"name": "AttenuateGroup", "params": {"group": 2, "factor": 0.3}},
                {"name": "WarmupCosineSchedule", "params": {"inner_steps": 65}},
            ],
            "description": "ClipGradNorm(1.0) -> AttenuateGroup(2, 0.3) -> WarmupCosineSchedule(65)",
        },
        {
            "id": "prog_seed909",
            "condition": "Learned_DSL_Programs",
            "family": "tool_workflow_chained_test",
            "seed": 909,
            "best_val_score": 1.1474,
            "ops": [
                {"name": "ClipGradNorm", "params": {"max_norm": 1.0}},
                {"name": "ProtectGroup", "params": {"group": 0}},
                {"name": "ReplayAnchorBatch", "params": {"ratio": 0.25}},
            ],
            "description": "ClipGradNorm(1.0) -> ProtectGroup(0) -> ReplayAnchorBatch(0.25)",
        },
    ]

    # DAG Nodes and directed Edges for lineage rendering
    nodes = [
        {"id": "gen0_root", "label": "Initial Heuristic", "type": "root", "val_score": 0.25, "generation": 0, "summary": "ClipGradNorm(1.0)"},
        {"id": "gen1_cand1", "label": "Cand 1.1", "type": "candidate", "val_score": 0.42, "generation": 1, "summary": "ClipGradNorm + Attenuate(group 1)"},
        {"id": "gen1_cand2", "label": "Cand 1.2", "type": "candidate", "val_score": 0.61, "generation": 1, "summary": "ClipGradNorm + Warmup(45)"},
        {"id": "gen2_cand1", "label": "Cand 2.1", "type": "candidate", "val_score": 0.88, "generation": 2, "summary": "ClipGradNorm + Attenuate(4, 0.3) + Warmup(55)"},
        {"id": "gen2_cand2", "label": "Cand 2.2", "type": "candidate", "val_score": 0.79, "generation": 2, "summary": "ClipGradNorm + Protect(group 0) + Replay(0.2)"},
        {"id": "gen3_best", "label": "Evolved Best (S1)", "type": "best", "val_score": 1.1995, "generation": 3, "summary": "ClipGradNorm(1.0) + Attenuate(2, 0.3) + Warmup(65)"},
    ]

    edges = [
        {"source": "gen0_root", "target": "gen1_cand1"},
        {"source": "gen0_root", "target": "gen1_cand2"},
        {"source": "gen1_cand2", "target": "gen2_cand1"},
        {"source": "gen1_cand1", "target": "gen2_cand2"},
        {"source": "gen2_cand1", "target": "gen3_best"},
    ]

    return {
        "operators": operators,
        "best_programs": best_programs,
        "nodes": nodes,
        "edges": edges,
    }
