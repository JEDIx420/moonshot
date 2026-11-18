#!/usr/bin/env python3
"""Poll one Kaggle kernel every ten minutes until terminal; preserve evidence.

Run from the authenticated Antigravity environment after kaggle kernels push.
This script never creates, reruns or overwrites a Kaggle kernel.
"""

import argparse
import csv
import datetime as dt
import json
from pathlib import Path
import subprocess
import time

DEFAULT_SLUG = "jayanandj/moonshot-cycle7-rift-intelligence"


def classify_status(raw):
    value = raw.upper()
    if any(word in value for word in ("FAILED", "ERROR", "CANCELLED", "CANCELED")):
        return "failed"
    if any(word in value for word in ("COMPLETE", "SUCCESS", "SUCCEEDED")):
        return "completed"
    if any(word in value for word in ("RUNNING", "EXECUTING", "STARTED")):
        return "running"
    if any(word in value for word in ("QUEUED", "PENDING", "WAITING")):
        return "queued"
    return "unknown"


def capture(args):
    result = subprocess.run(args, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=120, check=False)
    return result.returncode, result.stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", default=DEFAULT_SLUG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=int, default=600)
    parser.add_argument("--max-hours", type=float, default=48)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval_seconds < 60 and not args.once:
        parser.error("interval must be at least 60 seconds")
    if args.max_hours <= 0:
        parser.error("max-hours must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = args.output_dir / "status_history.tsv"
    first_write = not history.exists()
    deadline = time.monotonic() + 3600 * args.max_hours
    failures = 0
    with history.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, delimiter="\t")
        if first_write:
            writer.writerow(["utc", "kernel", "state", "status_exit_code", "status_output"])
        while True:
            now = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            try:
                code, status = capture(["kaggle", "kernels", "status", args.kernel])
                state = classify_status(status) if code == 0 else "api_error"
                failures = failures + 1 if code else 0
                writer.writerow([now, args.kernel, state, code,
                                 status.replace("\t", " ").replace("\n", " | ")])
                stream.flush()
                print(f"[{now}] {state}: {status}", flush=True)
                logs_code, logs = capture(["kaggle", "kernels", "logs", args.kernel])
                (args.output_dir / f"logs_{now}.txt").write_text(
                    f"CLI exit: {logs_code}\n{logs}\n", encoding="utf-8")
            except (OSError, subprocess.TimeoutExpired) as error:
                failures += 1
                state = "api_error"
                writer.writerow([now, args.kernel, state, -1, str(error)])
                stream.flush()
                print(f"[{now}] API error: {error}", flush=True)
            if state in ("completed", "failed"):
                result = "success" if state == "completed" else "kernel_failed"
                break
            if failures >= 3:
                result = "monitor_error"
                break
            if args.once:
                result = "snapshot_only"
                break
            if time.monotonic() >= deadline:
                result = "monitor_timeout_not_terminal"
                break
            time.sleep(args.interval_seconds)
    outcome = {"kernel": args.kernel, "outcome": result,
               "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
               "history_file": str(history)}
    (args.output_dir / "monitor_outcome.json").write_text(
        json.dumps(outcome, indent=2) + "\n", encoding="utf-8")
    if result == "success":
        print("Kernel completed; download artifacts and run independent review.")
    else:
        print(f"Monitoring stopped: {result}; inspect history and logs.")
    if result not in ("success", "snapshot_only"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
