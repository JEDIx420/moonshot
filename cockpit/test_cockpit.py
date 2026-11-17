#!/usr/bin/env python3
"""Automated Verification Test Suite for Moonshot Cockpit Dashboard.

Tests data loaders, streaming JSONL fault tolerance, synthetic mock generators,
and live HTTP endpoints on an active server instance.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
import urllib.request
import urllib.error

# Import server components
import server
import mock_data


class TestCockpitLoaders(unittest.TestCase):
    """Test data loader functions against real and synthetic files."""

    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parent.parent
        cls.experiments_dir = cls.repo_root / "experiments"

    def test_mock_data_generators(self):
        """Verify synthetic mock data integrity and schema compliance."""
        ledger = mock_data.get_mock_ledger()
        self.assertIsInstance(ledger, list)
        self.assertGreater(len(ledger), 0)
        self.assertIn("run_id", ledger[0])
        self.assertIn("status", ledger[0])
        self.assertIn("outcome", ledger[0])

        retention = mock_data.get_mock_retention()
        self.assertIn("conditions", retention)
        self.assertIn("tasks", retention)
        self.assertIn("matrix", retention)
        self.assertIn("1M_Rank4_Modular_Adapters", retention["conditions"])

        curves = mock_data.get_mock_curves()
        self.assertIn("meta_curve", curves)
        self.assertIn("wave1_curves", curves)
        self.assertGreater(len(curves["meta_curve"]), 0)

        lineage = mock_data.get_mock_lineage()
        self.assertIn("operators", lineage)
        self.assertIn("best_programs", lineage)
        self.assertIn("nodes", lineage)
        self.assertIn("edges", lineage)

    def test_load_real_run_ledger(self):
        """Verify real run_ledger.jsonl parsing."""
        if not self.experiments_dir.exists():
            self.skipTest("experiments directory not available")

        runs = server.load_run_ledger(self.experiments_dir, mock=False)
        self.assertIsInstance(runs, list)
        self.assertGreaterEqual(len(runs), 18)

        # Check latest run
        run_ids = [r["run_id"] for r in runs]
        self.assertIn("run_18_wave6b_1m_modular_subspace", run_ids)

        summary = server.compute_status_summary(runs)
        self.assertEqual(summary["total_runs"], len(runs))
        self.assertGreater(summary["total_compute_hours"], 0)

    def test_load_real_retention_data(self):
        """Verify real retention matrix CSV parsing."""
        if not self.experiments_dir.exists():
            self.skipTest("experiments directory not available")

        ret = server.load_retention_data(self.experiments_dir, mock=False)
        self.assertIn("conditions", ret)
        self.assertIn("tasks", ret)
        self.assertIn("matrix", ret)
        self.assertIn("1M_Rank4_Modular_Adapters", ret["conditions"])
        self.assertIn("ToolWorkflowBasic", ret["tasks"])

        # Check modular adapter retention vs full fine-tuning
        mod_val = ret["matrix"]["1M_Rank4_Modular_Adapters"]["ToolWorkflowBasic"]["retention_ratio"]
        fft_val = ret["matrix"]["1M_Full_FineTuning"]["ToolWorkflowBasic"]["retention_ratio"]
        self.assertGreater(mod_val, fft_val)

    def test_load_real_curves_data(self):
        """Verify real learning curves CSV parsing."""
        if not self.experiments_dir.exists():
            self.skipTest("experiments directory not available")

        curves = server.load_learning_curves(self.experiments_dir, mock=False)
        self.assertIn("meta_curve", curves)
        self.assertIn("wave1_curves", curves)
        self.assertGreater(len(curves["meta_curve"]), 0)
        self.assertIn("learned", curves["wave1_curves"])

    def test_load_real_ast_lineage(self):
        """Verify real AST program lineage from Wave 5e."""
        if not self.experiments_dir.exists():
            self.skipTest("experiments directory not available")

        lineage = server.load_ast_lineage(self.experiments_dir, mock=False)
        self.assertIn("operators", lineage)
        self.assertIn("best_programs", lineage)
        self.assertIn("nodes", lineage)
        self.assertIn("edges", lineage)
        self.assertGreaterEqual(len(lineage["operators"]), 7)
        self.assertGreater(len(lineage["best_programs"]), 0)

    def test_jsonl_fault_tolerance(self):
        """Verify incomplete trailing lines and corrupted records are skipped safely."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            corrupt_file = tmp_path / "run_ledger.jsonl"
            with open(corrupt_file, "w", encoding="utf-8") as f:
                f.write('{"run_id": "run_01", "status": "COMPLETE", "outcome": "PASS"}\n')
                f.write('{"run_id": "run_02", "status": "COMPLETE", "outcome": "PASS"}\n')
                f.write('{"run_id": "run_03", "status": "COMPLETE", ')  # Half-written trailing line

            runs = server.load_run_ledger(tmp_path, mock=False)
            self.assertEqual(len(runs), 2)
            self.assertEqual(runs[0]["run_id"], "run_01")
            self.assertEqual(runs[1]["run_id"], "run_02")


class TestCockpitHttpServer(unittest.TestCase):
    """Test live HTTP server endpoints and responses."""

    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parent.parent
        cls.experiments_dir = cls.repo_root / "experiments"
        cls.host = "127.0.0.1"
        cls.port = 8899

        # Start background server
        cls.server = server.CockpitServer(
            (cls.host, cls.port),
            server.CockpitRequestHandler,
            experiments_dir=cls.experiments_dir,
            mock_mode=False,
        )
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        time.sleep(0.5)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=1.0)

    def _get(self, path: str):
        url = f"http://{self.host}:{self.port}{path}"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5) as resp:
            status = resp.status
            content_type = resp.headers.get("Content-Type", "")
            data = resp.read()
            return status, content_type, data

    def test_dashboard_index(self):
        """Verify GET / serves index.html."""
        status, content_type, data = self._get("/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", content_type)
        html = data.decode("utf-8")
        self.assertIn("Moonshot Cockpit", html)
        self.assertIn("Live Retention Heatmaps", html)
        self.assertIn("AST Program Lineage", html)

    def test_api_health(self):
        """Verify GET /api/health returns 200 and status ok."""
        status, content_type, data = self._get("/api/health")
        self.assertEqual(status, 200)
        self.assertIn("application/json", content_type)
        payload = json.loads(data.decode("utf-8"))
        self.assertEqual(payload.get("status"), "ok")
        self.assertIn("timestamp", payload)

    def test_api_status(self):
        """Verify GET /api/status returns summary compute and runs."""
        status, content_type, data = self._get("/api/status")
        self.assertEqual(status, 200)
        payload = json.loads(data.decode("utf-8"))
        self.assertIn("total_runs", payload)
        self.assertIn("total_compute_hours", payload)
        self.assertGreaterEqual(payload["total_runs"], 18)

    def test_api_ledger(self):
        """Verify GET /api/ledger returns complete run list."""
        status, content_type, data = self._get("/api/ledger")
        self.assertEqual(status, 200)
        runs = json.loads(data.decode("utf-8"))
        self.assertIsInstance(runs, list)
        self.assertGreaterEqual(len(runs), 18)

        # Test query parameter filtering
        status, _, data_filtered = self._get("/api/ledger?status=COMPLETE")
        filtered_runs = json.loads(data_filtered.decode("utf-8"))
        for r in filtered_runs:
            self.assertEqual(r["status"], "COMPLETE")

    def test_api_retention(self):
        """Verify GET /api/retention returns 2D matrix data."""
        status, content_type, data = self._get("/api/retention")
        self.assertEqual(status, 200)
        payload = json.loads(data.decode("utf-8"))
        self.assertIn("matrix", payload)
        self.assertIn("conditions", payload)
        self.assertIn("tasks", payload)
        self.assertIn("1M_Rank4_Modular_Adapters", payload["conditions"])

    def test_api_curves(self):
        """Verify GET /api/curves returns training and meta curve series."""
        status, content_type, data = self._get("/api/curves")
        self.assertEqual(status, 200)
        payload = json.loads(data.decode("utf-8"))
        self.assertIn("meta_curve", payload)
        self.assertIn("wave1_curves", payload)
        self.assertGreater(len(payload["meta_curve"]), 0)

    def test_api_lineage(self):
        """Verify GET /api/lineage returns operators and evolved programs."""
        status, content_type, data = self._get("/api/lineage")
        self.assertEqual(status, 200)
        payload = json.loads(data.decode("utf-8"))
        self.assertIn("operators", payload)
        self.assertIn("best_programs", payload)
        self.assertIn("nodes", payload)
        self.assertIn("edges", payload)

    def test_api_not_found(self):
        """Verify unrecognized routes return 404."""
        url = f"http://{self.host}:{self.port}/api/non_existent_endpoint"
        try:
            urllib.request.urlopen(url, timeout=5)
            self.fail("Expected HTTPError 404")
        except urllib.error.HTTPError as err:
            self.assertEqual(err.code, 404)


if __name__ == "__main__":
    unittest.main()
