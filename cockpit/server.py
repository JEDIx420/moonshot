#!/usr/bin/env python3
"""Moonshot Cockpit Dashboard Server.

Pure Python standard-library HTTP server providing real-time telemetry,
retention heatmaps, learning curves, and AST program lineage visualization.
Zero external dependencies.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import logging
import os
from pathlib import Path
import sys
import threading
from typing import Any, Dict, List, Optional
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Optional mock data fallback
try:
    from cockpit import mock_data
except ImportError:
    import mock_data  # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("moonshot_cockpit")


def resolve_experiments_dir(candidate_path: Optional[str] = None) -> Path:
    """Resolve experiments directory path relative to current or project directory."""
    if candidate_path:
        p = Path(candidate_path).resolve()
        if p.exists():
            return p

    # Standard search locations
    repo_root = Path(__file__).resolve().parent.parent
    candidates = [
        Path.cwd() / "experiments",
        Path.cwd().parent / "experiments",
        repo_root / "experiments",
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            return c.resolve()

    return repo_root / "experiments"


def load_run_ledger(experiments_dir: Path, mock: bool = False) -> List[Dict[str, Any]]:
    """Parse experiments/run_ledger.jsonl with fault tolerance for incomplete writes."""
    if mock:
        return mock_data.get_mock_ledger()

    ledger_path = experiments_dir / "run_ledger.jsonl"
    if not ledger_path.exists():
        logger.warning(f"Ledger file not found at {ledger_path}, falling back to mock data")
        return mock_data.get_mock_ledger()

    runs: List[Dict[str, Any]] = []
    try:
        with open(ledger_path, "r", encoding="utf-8") as f:
            for line_no, raw_line in enumerate(f, 1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    runs.append(entry)
                except json.JSONDecodeError as err:
                    # Incomplete trailing line during active streaming write is ignored safely
                    logger.warning(f"Skipping malformed or incomplete JSONL line {line_no}: {err}")
    except Exception as exc:
        logger.error(f"Error reading {ledger_path}: {exc}")
        return mock_data.get_mock_ledger()

    if not runs:
        return mock_data.get_mock_ledger()

    return runs


def load_retention_data(experiments_dir: Path, mock: bool = False) -> Dict[str, Any]:
    """Parse retention matrix CSVs and compute aggregated condition x task matrix."""
    if mock:
        return mock_data.get_mock_retention()

    primary_csv = experiments_dir / "20261117-wave6b-1m-modular-subspace" / "retention_matrix_1m.csv"
    fallback_csv = experiments_dir / "20261116-wave5d-modular-adapters" / "retention_matrix.csv"

    target_csv = primary_csv if primary_csv.exists() else (fallback_csv if fallback_csv.exists() else None)
    if not target_csv:
        logger.warning("No retention matrix CSV found in experiments, using mock retention data")
        return mock_data.get_mock_retention()

    rows: List[Dict[str, Any]] = []
    matrix: Dict[str, Dict[str, Dict[str, float]]] = {}
    conditions_set = set()
    tasks_set = set()

    try:
        with open(target_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                cond = r.get("condition", "Unknown")
                task = r.get("eval_task", "Unknown")
                try:
                    retention_ratio = float(r.get("retention_ratio", 0.0))
                    exact_match = float(r.get("exact_match", 0.0))
                    token_acc = float(r.get("token_acc", 0.0))
                    seed = int(r.get("seed", 0))
                    stage = int(r.get("stage", 1))
                except (ValueError, TypeError):
                    continue

                item = {
                    "condition": cond,
                    "eval_task": task,
                    "retention_ratio": retention_ratio,
                    "exact_match": exact_match,
                    "token_acc": token_acc,
                    "seed": seed,
                    "stage": stage,
                }
                rows.append(item)
                conditions_set.add(cond)
                tasks_set.add(task)

                if cond not in matrix:
                    matrix[cond] = {}
                if task not in matrix[cond]:
                    matrix[cond][task] = {
                        "retention_ratio": retention_ratio,
                        "exact_match": exact_match,
                        "token_acc": token_acc,
                        "samples_count": 1,
                    }
                else:
                    # Average over seeds/stages
                    curr = matrix[cond][task]
                    cnt = curr["samples_count"] + 1
                    curr["retention_ratio"] = round((curr["retention_ratio"] * curr["samples_count"] + retention_ratio) / cnt, 4)
                    curr["exact_match"] = round((curr["exact_match"] * curr["samples_count"] + exact_match) / cnt, 4)
                    curr["token_acc"] = round((curr["token_acc"] * curr["samples_count"] + token_acc) / cnt, 4)
                    curr["samples_count"] = cnt
    except Exception as exc:
        logger.error(f"Error parsing retention CSV {target_csv}: {exc}")
        return mock_data.get_mock_retention()

    # Sort conditions and tasks deterministically
    sorted_conditions = sorted(list(conditions_set))
    sorted_tasks = sorted(list(tasks_set))

    return {
        "dataset_name": target_csv.name,
        "dataset_path": str(target_csv),
        "conditions": sorted_conditions,
        "tasks": sorted_tasks,
        "matrix": matrix,
        "rows": rows,
    }


def load_learning_curves(experiments_dir: Path, mock: bool = False) -> Dict[str, Any]:
    """Parse learning curves from meta_curve.csv and wave1 learning_curves.csv."""
    if mock:
        return mock_data.get_mock_curves()

    meta_curve_path = experiments_dir / "20261116-wave5c-learned-vs-random" / "meta_curve.csv"
    wave1_dir = experiments_dir / "20261116-wave1-learned-editor"

    meta_trials: List[Dict[str, Any]] = []
    if meta_curve_path.exists():
        try:
            with open(meta_curve_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    try:
                        meta_trials.append({
                            "trial": int(r.get("trial", 0)),
                            "family": r.get("family", ""),
                            "raw_reward": float(r.get("raw_reward", 0.0)),
                            "running_baseline": float(r.get("running_baseline", 0.0)),
                            "learned_loss": float(r.get("learned_loss", 0.0)),
                            "shuffled_reward": float(r.get("shuffled_reward", 0.0)),
                        })
                    except (ValueError, TypeError):
                        continue
        except Exception as exc:
            logger.error(f"Error reading meta_curve.csv: {exc}")

    wave1_series: Dict[str, List[Dict[str, Any]]] = {}
    conditions = ["learned", "shuffled", "heuristic", "frozen", "random"]
    for cond in conditions:
        cond_csv = wave1_dir / cond / "learning_curves.csv"
        if cond_csv.exists():
            series = []
            try:
                with open(cond_csv, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for r in reader:
                        try:
                            series.append({
                                "trial": int(r.get("trial_idx", 0)),
                                "family": r.get("task_family", ""),
                                "candidate_id": r.get("candidate_id", ""),
                                "status": r.get("status", ""),
                                "acc": float(r.get("final_exact_acc", 0.0)),
                                "gpu_secs": float(r.get("gpu_secs", 0.0)),
                            })
                        except (ValueError, TypeError):
                            continue
                wave1_series[cond] = series
            except Exception as exc:
                logger.error(f"Error reading {cond_csv}: {exc}")

    if not meta_trials and not wave1_series:
        return mock_data.get_mock_curves()

    return {
        "meta_curve": meta_trials,
        "wave1_curves": wave1_series,
    }


def load_ast_lineage(experiments_dir: Path, mock: bool = False) -> Dict[str, Any]:
    """Parse AST program lineage and policy programs from Wave 5e."""
    if mock:
        return mock_data.get_mock_lineage()

    dsl_dir = experiments_dir / "20261116-wave5e-update-rule-dsl"
    schema_path = dsl_dir / "update_rule_schema.json"
    lineage_path = dsl_dir / "program_lineage.jsonl"
    policy_path = dsl_dir / "policy_programs.jsonl"

    operators: List[Dict[str, Any]] = []
    if schema_path.exists():
        try:
            with open(schema_path, "r", encoding="utf-8") as f:
                schema_data = json.load(f)
                operators = schema_data.get("operators", [])
        except Exception as exc:
            logger.error(f"Error reading {schema_path}: {exc}")

    best_programs: List[Dict[str, Any]] = []
    if lineage_path.exists():
        try:
            with open(lineage_path, "r", encoding="utf-8") as f:
                for line_idx, raw_line in enumerate(f, 1):
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        ops_raw = entry.get("best_program", {}).get("ops", [])
                        ops_formatted = []
                        desc_parts = []
                        for op in ops_raw:
                            if isinstance(op, dict):
                                for op_name, op_val in op.items():
                                    ops_formatted.append({"name": op_name, "params": op_val})
                                    desc_parts.append(f"{op_name}({op_val})")
                            else:
                                ops_formatted.append({"name": str(op), "params": None})
                                desc_parts.append(str(op))

                        prog_obj = {
                            "id": f"prog_{entry.get('seed', line_idx)}_{line_idx}",
                            "condition": entry.get("condition", "Learned_DSL_Programs"),
                            "family": entry.get("family", ""),
                            "seed": entry.get("seed", 0),
                            "best_val_score": round(float(entry.get("best_val_score", 0.0)), 4),
                            "ops": ops_formatted,
                            "description": " -> ".join(desc_parts) if desc_parts else "IdentityNoOp",
                        }
                        best_programs.append(prog_obj)
                    except Exception:
                        continue
        except Exception as exc:
            logger.error(f"Error reading {lineage_path}: {exc}")

    # Build DAG nodes and edges from lineage and policy trials
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    if best_programs:
        # Create Root Node
        nodes.append({
            "id": "node_root",
            "label": "DSL Root (Identity)",
            "type": "root",
            "val_score": 0.0,
            "generation": 0,
            "summary": "IdentityNoOp",
        })

        prev_node_id = "node_root"
        for idx, prog in enumerate(best_programs, 1):
            node_id = f"node_{prog['id']}"
            nodes.append({
                "id": node_id,
                "label": f"Program Seed {prog['seed']} ({prog['family'][:14]}...)",
                "type": "best" if prog["best_val_score"] > 1.0 else "candidate",
                "val_score": prog["best_val_score"],
                "generation": idx,
                "summary": prog["description"],
                "seed": prog["seed"],
                "family": prog["family"],
            })
            edges.append({
                "source": prev_node_id,
                "target": node_id,
            })
            # Connect to root as base reference as well
            if prev_node_id != "node_root":
                edges.append({
                    "source": "node_root",
                    "target": node_id,
                })
            prev_node_id = node_id

    if not operators and not best_programs:
        return mock_data.get_mock_lineage()

    return {
        "operators": operators,
        "best_programs": best_programs,
        "nodes": nodes,
        "edges": edges,
    }


def compute_status_summary(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute high-level summary telemetry from run ledger."""
    total_runs = len(runs)
    completed = sum(1 for r in runs if r.get("status") == "COMPLETE")
    total_cpu_mins = sum(float(r.get("cpu_minutes", 0.0) or 0.0) for r in runs)
    total_gpu_mins = sum(float(r.get("gpu_minutes", 0.0) or 0.0) for r in runs)

    latest_run = runs[-1] if runs else {}
    return {
        "total_runs": total_runs,
        "completed_runs": completed,
        "total_cpu_hours": round(total_cpu_mins / 60.0, 2),
        "total_gpu_hours": round(total_gpu_mins / 60.0, 2),
        "total_compute_hours": round((total_cpu_mins + total_gpu_mins) / 60.0, 2),
        "latest_run_id": latest_run.get("run_id", "N/A"),
        "latest_outcome": latest_run.get("outcome", "N/A"),
        "latest_timestamp": latest_run.get("timestamp", "N/A"),
    }


class CockpitRequestHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler serving Cockpit SPA and REST API endpoints."""

    server_version = "MoonshotCockpit/1.0"

    def do_HEAD(self) -> None:
        """Handle HEAD requests by delegating headers without body."""
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        if path in ("/", "/index.html"):
            html_path = Path(__file__).resolve().parent / "index.html"
            if html_path.exists():
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(html_path.stat().st_size))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

    def do_OPTIONS(self) -> None:
        """Handle CORS preflight requests."""
        self.send_response(HTTPStatus.OK)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_GET(self) -> None:
        """Route GET requests to API endpoints or static UI."""
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        query_params = urllib.parse.parse_qs(parsed_url.query)

        # Static Dashboard Page
        if path in ("/", "/index.html"):
            self._serve_dashboard()
            return

        # Health Endpoint
        if path == "/api/health":
            self._send_json({
                "status": "ok",
                "mock_mode": self.server.mock_mode,  # type: ignore[attr-defined]
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "experiments_dir": str(self.server.experiments_dir),  # type: ignore[attr-defined]
            })
            return

        # Overall Status Summary Endpoint
        if path == "/api/status":
            runs = load_run_ledger(self.server.experiments_dir, self.server.mock_mode)  # type: ignore[attr-defined]
            summary = compute_status_summary(runs)
            self._send_json(summary)
            return

        # Run Ledger Endpoint (/api/ledger or /api/runs)
        if path in ("/api/ledger", "/api/runs"):
            runs = load_run_ledger(self.server.experiments_dir, self.server.mock_mode)  # type: ignore[attr-defined]
            # Optional query filters
            if "status" in query_params:
                st = query_params["status"][0].upper()
                runs = [r for r in runs if r.get("status", "").upper() == st]
            if "search" in query_params:
                q = query_params["search"][0].lower()
                runs = [
                    r for r in runs
                    if q in r.get("run_id", "").lower()
                    or q in r.get("outcome", "").lower()
                    or q in r.get("finding", "").lower()
                ]
            if "limit" in query_params:
                try:
                    limit = int(query_params["limit"][0])
                    runs = runs[-limit:]
                except ValueError:
                    pass
            self._send_json(runs)
            return

        # Retention Heatmap Data Endpoint
        if path == "/api/retention":
            data = load_retention_data(self.server.experiments_dir, self.server.mock_mode)  # type: ignore[attr-defined]
            self._send_json(data)
            return

        # Learning Curves Data Endpoint
        if path == "/api/curves":
            data = load_learning_curves(self.server.experiments_dir, self.server.mock_mode)  # type: ignore[attr-defined]
            self._send_json(data)
            return

        # AST Program Lineage Data Endpoint
        if path == "/api/lineage":
            data = load_ast_lineage(self.server.experiments_dir, self.server.mock_mode)  # type: ignore[attr-defined]
            self._send_json(data)
            return

        # 404 For Unrecognized Endpoints
        self._send_json({"error": "Not Found", "path": path}, status=HTTPStatus.NOT_FOUND)

    def _serve_dashboard(self) -> None:
        """Serve the cockpit index.html file."""
        html_path = Path(__file__).resolve().parent / "index.html"
        if not html_path.exists():
            self._send_json(
                {"error": "index.html not found on server"},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return

        try:
            with open(html_path, "rb") as f:
                content = f.read()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(content)
        except Exception as exc:
            logger.error(f"Error serving index.html: {exc}")
            self._send_json(
                {"error": f"Failed to read index.html: {exc}"},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def _send_json(self, data: Any, status: int = HTTPStatus.OK) -> None:
        """Serialize data to JSON with appropriate headers and status."""
        try:
            encoded = json.dumps(data, indent=2, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(encoded)
        except Exception as exc:
            logger.error(f"Error sending JSON response: {exc}")

    def log_message(self, format: str, *args: Any) -> None:
        """Log concise HTTP access requests."""
        logger.info(f"{self.client_address[0]} - {format % args}")


class CockpitServer(ThreadingHTTPServer):
    """Threading HTTP Server holding application configuration state."""

    def __init__(
        self,
        server_address: tuple[str, int],
        RequestHandlerClass: type[BaseHTTPRequestHandler],
        experiments_dir: Path,
        mock_mode: bool = False,
    ):
        super().__init__(server_address, RequestHandlerClass)
        self.experiments_dir = experiments_dir
        self.mock_mode = mock_mode


def main() -> None:
    """CLI entrypoint for Moonshot Cockpit Server."""
    parser = argparse.ArgumentParser(
        description="Moonshot Cockpit Dashboard Server (Zero External Dependencies)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Server binding host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server binding port (default: 8000)",
    )
    parser.add_argument(
        "--experiments-dir",
        type=str,
        default=None,
        help="Path to experiments directory (default: auto-detected)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in synthetic mock data mode",
    )
    args = parser.parse_args()

    exp_dir = resolve_experiments_dir(args.experiments_dir)
    logger.info(f"Starting Moonshot Cockpit Dashboard on http://{args.host}:{args.port}")
    logger.info(f"Experiments Directory: {exp_dir} (Exists: {exp_dir.exists()})")
    logger.info(f"Mock Mode: {args.mock}")

    server = CockpitServer(
        (args.host, args.port),
        CockpitRequestHandler,
        experiments_dir=exp_dir,
        mock_mode=args.mock,
    )

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down Moonshot Cockpit server...")
    finally:
        server.shutdown()
        server.server_close()
        logger.info("Cockpit server terminated cleanly.")


if __name__ == "__main__":
    main()
