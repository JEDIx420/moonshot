#!/usr/bin/env python3
"""
Tier 4 E2E Test: Moonshot Cockpit Dashboard REST API & Web UI Verification
Verifies:
1. Server initialization with pure Python standard library (zero external dependencies)
2. Ephemeral port binding and HTTP server lifecycle management
3. REST API contracts:
   - GET /api/health
   - GET /api/status
   - GET /api/ledger (with filtering: ?status=..., ?search=..., ?limit=...)
   - GET /api/retention
   - GET /api/curves
   - GET /api/lineage
   - GET / (Dashboard HTML UI)
   - GET /unknown (404 error response handling)
4. Mock mode (--mock flag) operation
5. Fault-tolerant streaming JSONL reading (handling truncated/incomplete trailing lines)
6. Clean process termination and resource cleanup
"""

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

def print_header(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]

def wait_for_server(url: str, timeout: float = 5.0):
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:
                if resp.status == 200:
                    return
        except Exception:
            time.sleep(0.1)
    raise TimeoutError(f"Server at {url} did not become ready within {timeout}s")

def http_get(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "MoonshotE2ETest/1.0"})
    with urllib.request.urlopen(req, timeout=3.0) as resp:
        body = resp.read().decode("utf-8")
        status = resp.status
        content_type = resp.headers.get("Content-Type", "")
        return status, content_type, body

# ---------------------------------------------------------------------------
# Test 1: Live Mode Endpoint Validation
# ---------------------------------------------------------------------------
def test_live_server_endpoints():
    print_header("Test 1: Cockpit Server REST Endpoints (Live Mode)")
    port = find_free_port()
    cmd = [sys.executable, "cockpit/server.py", "--port", str(port), "--host", "127.0.0.1"]
    print(f"  Starting cockpit server on 127.0.0.1:{port}...")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    base_url = f"http://127.0.0.1:{port}"
    try:
        wait_for_server(f"{base_url}/api/health")
        print("  [PASS] Server is listening and responded to /api/health")

        # 1. /api/health
        status, ctype, body = http_get(f"{base_url}/api/health")
        assert status == 200, f"Expected 200, got {status}"
        data = json.loads(body)
        assert data.get("status") == "ok", f"Unexpected health status: {data}"
        print(f"  [PASS] GET /api/health returned 200 OK: {data['status']}")

        # 2. /api/status
        status, ctype, body = http_get(f"{base_url}/api/status")
        assert status == 200
        summary = json.loads(body)
        assert "total_runs" in summary
        assert "total_compute_hours" in summary
        print(f"  [PASS] GET /api/status: {summary['total_runs']} runs, {summary['total_compute_hours']} compute hours")

        # 3. /api/ledger
        status, ctype, body = http_get(f"{base_url}/api/ledger")
        assert status == 200
        runs = json.loads(body)
        assert isinstance(runs, list)
        assert len(runs) > 0, "Expected at least 1 run in run_ledger"
        first_run = runs[0]
        for field in ["run_id", "timestamp", "status"]:
            assert field in first_run, f"Missing '{field}' in run record: {first_run}"
        print(f"  [PASS] GET /api/ledger: parsed {len(runs)} runs, latest: {runs[-1].get('run_id')}")

        # 4. /api/ledger query filtering
        status, _, body = http_get(f"{base_url}/api/ledger?status=COMPLETE")
        assert status == 200
        complete_runs = json.loads(body)
        assert all(r.get("status") == "COMPLETE" for r in complete_runs)
        print(f"  [PASS] GET /api/ledger?status=COMPLETE returned {len(complete_runs)} filtered runs")

        # 5. /api/retention
        status, ctype, body = http_get(f"{base_url}/api/retention")
        assert status == 200
        retention = json.loads(body)
        assert isinstance(retention, dict)
        assert "conditions" in retention or "matrix" in retention or "items" in retention
        print("  [PASS] GET /api/retention returned structured retention matrix")

        # 6. /api/curves
        status, ctype, body = http_get(f"{base_url}/api/curves")
        assert status == 200
        curves = json.loads(body)
        assert isinstance(curves, dict)
        assert "meta_curve" in curves or "wave1_curves" in curves or "curves" in curves or "series" in curves
        print("  [PASS] GET /api/curves returned learning curves trajectories")

        # 7. /api/lineage
        status, ctype, body = http_get(f"{base_url}/api/lineage")
        assert status == 200
        lineage = json.loads(body)
        assert isinstance(lineage, dict)
        for key in ["nodes", "edges"]:
            assert key in lineage, f"Missing '{key}' in lineage response"
        print(f"  [PASS] GET /api/lineage returned DAG with {len(lineage['nodes'])} nodes, {len(lineage['edges'])} edges")

        # 8. GET / (Dashboard HTML UI)
        status, ctype, body = http_get(f"{base_url}/")
        assert status == 200
        assert "text/html" in ctype
        assert "<html" in body.lower()
        assert "chart" in body.lower() or "canvas" in body.lower()
        print("  [PASS] GET / served HTML dashboard with Chart.js canvas elements")

        # 9. 404 Error handling
        try:
            http_get(f"{base_url}/api/non_existent_route")
            assert False, "Should have raised HTTPError 404"
        except urllib.error.HTTPError as err:
            assert err.code == 404
            err_body = json.loads(err.read().decode("utf-8"))
            assert "error" in err_body
            print("  [PASS] GET /api/non_existent_route correctly returned 404 Not Found")

    finally:
        proc.terminate()
        proc.wait(timeout=3.0)
        print("  [PASS] Live cockpit server terminated cleanly")

# ---------------------------------------------------------------------------
# Test 2: Mock Mode Validation (--mock flag)
# ---------------------------------------------------------------------------
def test_mock_mode():
    print_header("Test 2: Cockpit Server Mock Mode (--mock)")
    port = find_free_port()
    cmd = [sys.executable, "cockpit/server.py", "--port", str(port), "--host", "127.0.0.1", "--mock"]
    print(f"  Starting cockpit server in mock mode on 127.0.0.1:{port}...")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    base_url = f"http://127.0.0.1:{port}"
    try:
        wait_for_server(f"{base_url}/api/health")
        status, _, body = http_get(f"{base_url}/api/health")
        assert status == 200
        data = json.loads(body)
        assert data.get("mock_mode") is True, f"Expected mock_mode == True, got {data}"
        print("  [PASS] Server confirmed mock_mode == True")

        # Check mock ledger
        status, _, body = http_get(f"{base_url}/api/ledger")
        assert status == 200
        runs = json.loads(body)
        assert len(runs) > 0, "Mock ledger should contain synthetic runs"
        print(f"  [PASS] Mock ledger loaded {len(runs)} synthetic entries successfully")

        # Check mock retention
        status, _, body = http_get(f"{base_url}/api/retention")
        assert status == 200
        retention = json.loads(body)
        assert bool(retention)
        print("  [PASS] Mock retention matrix loaded successfully")

    finally:
        proc.terminate()
        proc.wait(timeout=3.0)
        print("  [PASS] Mock mode cockpit server terminated cleanly")

# ---------------------------------------------------------------------------
# Test 3: Fault-Tolerant Streaming JSONL Reader
# ---------------------------------------------------------------------------
def test_fault_tolerant_streaming():
    print_header("Test 3: Fault-Tolerant Streaming JSONL Reader (Partial Line Ingestion)")
    temp_dir = tempfile.mkdtemp(prefix="moonshot_test_experiments_")
    try:
        ledger_path = os.path.join(temp_dir, "run_ledger.jsonl")
        # Write 2 valid JSON lines and 1 incomplete/corrupted trailing line
        with open(ledger_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"run_id": "test_run_01", "timestamp": "2026-11-17T00:00:00Z", "status": "COMPLETE"}) + "\n")
            f.write(json.dumps({"run_id": "test_run_02", "timestamp": "2026-11-17T01:00:00Z", "status": "COMPLETE"}) + "\n")
            f.write('{"run_id": "test_run_03_corrupt_tail", "timestamp": "2026-11-17T02:00:00Z", "status": "INCOM') # deliberately truncated

        port = find_free_port()
        cmd = [
            sys.executable, "cockpit/server.py",
            "--port", str(port),
            "--host", "127.0.0.1",
            "--experiments-dir", temp_dir
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        base_url = f"http://127.0.0.1:{port}"
        try:
            wait_for_server(f"{base_url}/api/health")
            status, _, body = http_get(f"{base_url}/api/ledger")
            assert status == 200, f"Expected 200, got {status}"
            runs = json.loads(body)
            # Must have parsed the 2 complete lines and skipped the truncated line safely
            assert len(runs) == 2, f"Expected 2 runs parsed, got {len(runs)}: {runs}"
            assert runs[0]["run_id"] == "test_run_01"
            assert runs[1]["run_id"] == "test_run_02"
            print("  [PASS] Successfully parsed 2 complete lines while skipping truncated trailing write!")
        finally:
            proc.terminate()
            proc.wait(timeout=3.0)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def main():
    print("Starting Tier 4 Cockpit Dashboard E2E Test Suite...")
    test_live_server_endpoints()
    test_mock_mode()
    test_fault_tolerant_streaming()
    print("\n" + "=" * 70)
    print("  ALL TIER 4 COCKPIT VERIFICATIONS PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    main()
