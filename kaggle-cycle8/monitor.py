#!/usr/bin/env python3
"""Watch one Cycle 8 Kaggle kernel every 600 seconds until terminal.

Does not create, cancel, resubmit, or overwrite a Kaggle job.
"""

import argparse
import csv
import datetime as dt
import json
from pathlib import Path
import subprocess
import time

KERNEL = "jayanandj/moonshot-cycle8-rift-evidence"


def classify_status(raw):
    val = raw.upper()
    if any(w in val for w in ("FAILED", "ERROR", "CANCELLED", "CANCELED")):
        return "failed"
    if any(w in val for w in ("COMPLETE", "SUCCESS", "SUCCEEDED")):
        return "completed"
    if any(w in val for w in ("RUNNING", "EXECUTING", "STARTED")):
        return "running"
    if any(w in val for w in ("QUEUED", "PENDING", "WAITING")):
        return "queued"
    return "unknown"


def capture(args):
    r = subprocess.run(args, text=True, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, timeout=120, check=False)
    return r.returncode, r.stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", default=KERNEL)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--interval-seconds", type=int, default=600)
    parser.add_argument("--max-hours", type=float, default=72)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if not args.once and args.interval_seconds < 60:
        parser.error("interval must be >=60s")
    if args.max_hours <= 0:
        parser.error("max-hours must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = args.output_dir / "status_history.tsv"
    deadline = time.monotonic() + args.max_hours * 3600
    failures = 0
    with history.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        if f.tell() == 0:
            writer.writerow(["utc", "kernel", "state", "cli_exit", "details"])
        while True:
            stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            try:
                code, status = capture(["kaggle", "kernels", "status", args.kernel])
                state = classify_status(status) if code == 0 else "api_error"
                failures = 0 if code == 0 else failures + 1
                writer.writerow([stamp, args.kernel, state, code,
                                 status.replace("\t", " ").replace("\n", " | ")])
                f.flush()
                print(f"[{stamp}] {state}: {status}", flush=True)
                log_code, log_text = capture(["kaggle", "kernels", "logs", args.kernel])
                (args.output_dir / f"logs_{stamp}.txt").write_text(
                    f"cli_exit={log_code}\n{log_text}\n", encoding="utf-8")
            except (OSError, subprocess.TimeoutExpired) as error:
                failures += 1
                state = "api_error"
                writer.writerow([stamp, args.kernel, state, -1, str(error)])
                f.flush()
            if state in ("completed", "failed"):
                outcome = state
                break
            if failures >= 3:
                outcome = "monitor_api_failure_not_terminal"
                break
            if args.once:
                outcome = "snapshot_only"
                break
            if time.monotonic() >= deadline:
                outcome = "monitor_deadline_not_terminal"
                break
            time.sleep(args.interval_seconds)
    (args.output_dir / "monitor_outcome.json").write_text(
        json.dumps({"kernel": args.kernel, "outcome": outcome, "history": str(history),
                    "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat()},
                   indent=2) + "\n", encoding="utf-8")
    if outcome not in ("completed", "snapshot_only"):
        raise SystemExit(f"Cycle 8 monitoring ended: {outcome}; investigate before resubmit")


if __name__ == "__main__":
    main()
