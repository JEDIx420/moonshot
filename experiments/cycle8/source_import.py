#!/usr/bin/env python3
"""Prepare unlabelled candidate drafts from actual *recorded* agent runs.

Does not create truth labels and is not a substitute for independent review.
Raw public benchmark transcripts may contain sensitive-looking content; keep
both inputs and drafts in private local storage, not Moonshot's git history.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_of(value, limit=1400):
    if value is None:
        return ""
    if isinstance(value, str):
        return value[:limit]
    return json.dumps(value, ensure_ascii=False, sort_keys=True)[:limit]


def sessions(data, mode):
    if isinstance(data, dict):
        if isinstance(data.get("messages"), list):
            yield data
        elif mode == "normalized":
            # Add a reviewed and tested adapter for any other public format.
            for name in ("trajectories", "results", "runs"):
                if isinstance(data.get(name), list):
                    for item in data[name]:
                        yield from sessions(item, mode)
    elif isinstance(data, list):
        for item in data:
            yield from sessions(item, mode)


def normalized_tool_events(messages):
    # In AgentDojo the tool call is echoed on tool-result messages, with
    # tool_call_id/tool_call.function. Never invent those fields.
    calls = {}
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        for call in msg.get("tool_calls") or []:
            if isinstance(call, dict) and call.get("id"):
                calls[call["id"]] = call
    events = []
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        call_id = msg.get("tool_call_id")
        call = msg.get("tool_call") or calls.get(call_id) or {}
        if not isinstance(call, dict):
            call = {}
        name = call.get("function") or msg.get("name") or msg.get("tool_name")
        if not isinstance(name, str) or not name.strip():
            continue
        error = msg.get("error")
        status = "error" if error else "ok" if msg.get("content") is not None else "unknown"
        events.append({"id": f"e{len(events)}", "tool_name": name.strip(),
                       "call_id": call_id if isinstance(call_id, str) else None,
                       "status": status,
                       "text": text_of(msg.get("content"))})
    return events


def final_text(messages):
    # A final answer *after* the last observed tool result. Don't extract
    # thinking messages or fabricate the agent's assertion from a tool result.
    tool_indices = [i for i, m in enumerate(messages) if isinstance(m, dict) and m.get("role") == "tool"]
    if not tool_indices:
        return ""
    for msg in reversed(messages[tool_indices[-1] + 1:]):
        if isinstance(msg, dict) and msg.get("role") == "assistant" and not msg.get("tool_calls"):
            raw = msg.get("content")
            if isinstance(raw, str):
                cleaned = re.sub(r"<thinking>.*?</thinking>", "", raw, flags=re.S | re.I).strip()
                return cleaned[:4000]
    return ""


def source_group(data, rel, mode):
    if mode == "agentdojo":
        suite, task = data.get("suite_name"), data.get("user_task_id")
        if not suite or not task:
            raise ValueError(f"missing AgentDojo task group for {rel}")
        return f"{suite}/{task}"
    task = data.get("task_id") or data.get("user_task_id") or data.get("id")
    if task is None:
        raise ValueError(f"missing normalized task_id in {rel}; no synthetic grouping")
    return str(task)


def extract(root, source, repo, revision, license_id, mode="agentdojo", max_files=0):
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("pin upstream to full 40-character Git commit")
    result, stats = [], Counter()
    paths = sorted(root.rglob("*.json"))
    if max_files:
        paths = paths[:max_files]
    for file in paths:
        stats["source_files"] += 1
        rel = file.relative_to(root).as_posix()
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            stats["invalid_json"] += 1
            continue
        source_sha = sha256(file)
        found = False
        for index, session in enumerate(sessions(data, mode)):
            found = True
            stats["sessions_seen"] += 1
            messages = session["messages"]
            if not isinstance(messages, list):
                continue
            events = normalized_tool_events(messages)
            final = final_text(messages)
            if len(events) < 2 or not final:
                stats["no_usable_tool_trace"] += 1
                continue
            try:
                group = source_group(session, rel, mode)
            except ValueError:
                stats["missing_task_group"] += 1
                continue
            # Preserve only the latest 16 observations in the draft; a
            # reviewer must verify whether the true grounding was truncated.
            selected = events[-16:]
            for i, event in enumerate(selected):
                event["id"] = f"e{i}"
            case_key = f"{source}|{rel}|{index}|{group}".encode()
            case_id = "trace_" + hashlib.sha256(case_key).hexdigest()[:24]
            result.append({"case_id": case_id, "source_id": source,
                           "group_id": group, "stratum": "natural",
                           "claim_text": "", "claim_call_id": None,
                           "assistant_final_text_for_review": final,
                           "candidate_truncated": len(events) > 16,
                           "candidates": selected,
                           "provenance": {"repo": repo, "revision": revision,
                                          "relative_path": rel, "sha256": source_sha,
                                          "license": license_id, "environment": "sandbox"},
                           "gold": None})
            stats["unlabelled_drafts"] += 1
        if not found:
            stats["unrecognized_source_schema"] += 1
    if not result:
        raise ValueError("no usable recorded tool trajectories; inspect source adapter, do not invent gold")
    return result, dict(stats)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--license", required=True)
    parser.add_argument("--format", choices=("agentdojo", "normalized"), default="agentdojo")
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.max_files < 0:
        parser.error("max-files must be nonnegative")
    try:
        local_revision = subprocess.check_output(
            ["git", "-C", str(args.source_root), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.PIPE).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("recorded source must be checked out from a pinned git revision") from error
    if local_revision != args.revision:
        raise ValueError(f"source checkout SHA is {local_revision}, expected {args.revision}")
    drafts, stats = extract(args.source_root, args.source_id, args.repo,
                            args.revision, args.license, args.format, args.max_files)
    if args.output.exists():
        raise ValueError("output exists; never overwrite an annotation draft")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for row in drafts:
            output.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "unlabelled_only": True,
                      "stats": stats}, indent=2))


if __name__ == "__main__":
    main()
