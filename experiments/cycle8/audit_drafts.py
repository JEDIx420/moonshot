#!/usr/bin/env python3
"""Check unlabelled drafts against pinned raw observations without exporting text."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess

from benchmark import PINNED_SOURCES, group_split, jsonl


def check(ok, reason):
    if not ok:
        raise ValueError(reason)


def audit(path, source_root, cohort):
    repo, revision = PINNED_SOURCES[cohort]
    git = ["git", "-C", str(source_root)]
    check(subprocess.check_output(git + ["rev-parse", "HEAD"], text=True).strip() == revision,
          "upstream revision mismatch")
    check(not subprocess.check_output(git + ["status", "--porcelain", "--untracked-files=all"], text=True),
          "upstream checkout is dirty")
    groups = set()
    case_ids = set()
    checked = Counter()
    files = {}
    license_path = Path(subprocess.check_output(git + ["rev-parse", "--show-toplevel"], text=True).strip()) / "LICENSE"
    license_bytes = license_path.read_bytes()
    check(b"MIT License" in license_bytes or b"Permission is hereby granted" in license_bytes,
          "MIT license text missing")
    for draft in jsonl(path):
        prov = draft["provenance"]
        case_id = draft["case_id"]
        check(draft["source_id"] == cohort and prov["repo"] == repo and prov["revision"] == revision,
              f"wrong pinned source at {case_id}")
        check(prov["environment"] == "sandbox" and prov["license"] == "MIT", f"bad scope at {case_id}")
        check(case_id not in case_ids and not draft["claim_text"] and draft["gold"] is None,
              f"duplicate or unexpected label at {case_id}")
        case_ids.add(case_id)
        relative = prov["relative_path"]
        raw_path = (source_root / relative).resolve()
        check(raw_path.is_relative_to(source_root.resolve()), "raw path escapes source root")
        if relative not in files:
            data = raw_path.read_bytes()
            files[relative] = (hashlib.sha256(data).hexdigest(), json.loads(data))
        digest, raw = files[relative]
        check(prov["sha256"] == digest, f"source bytes differ at {case_id}")
        if cohort == "tau_bench":
            session = raw[prov["session_index"]]
            domain = re.search(r"-(airline|retail)\.json$", relative)[1]
            group = f"{domain}/{session['task_id']}"
            messages = session["traj"]
        else:
            check(raw.get("attack_type") in (None, "none") and not raw.get("injections"),
                  f"injection variant in natural draft {case_id}")
            group = f"{raw['suite_name']}/{raw['user_task_id']}"
            messages = raw["messages"]
        check(group == draft["group_id"], f"task identity mismatch at {case_id}")
        groups.add(group)
        message_index = draft["assistant_final_message_index"]
        message = messages[message_index]
        check(message["role"] == "assistant" and not message.get("tool_calls") and
              draft["assistant_final_text_for_review"] in message["content"], f"claim context mismatch at {case_id}")
        check(2 <= len(draft["candidates"]) <= 16, f"candidate count at {case_id}")
        for event in draft["candidates"]:
            index = event["message_index"]
            raw_event = messages[index]
            content = raw_event["content"]
            encoded = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, sort_keys=True)
            check(index < message_index and raw_event["role"] == "tool" and
                  event["call_id"] == raw_event.get("tool_call_id") and encoded.startswith(event["text"]),
                  f"candidate provenance differs at {case_id}")
            checked["candidate_pointers_checked"] += 1
        checked["drafts_checked"] += 1
        checked["candidate_lists_truncated"] += int(draft["candidate_truncated"])
    check(case_ids, "no draft rows audited")
    return {"cohort": cohort, "repo": repo, "revision": revision,
            "license": "MIT", "license_sha256": hashlib.sha256(license_bytes).hexdigest(),
            "privacy": "raw text remains private locally; derived benchmark privacy review pending",
            "task_groups": len(groups), "raw_files_referenced": len(files),
            "raw_file_sha256": {name: value[0] for name, value in files.items()},
            "potential_group_split_counts": dict(Counter(group_split(cohort, group) for group in groups)),
            **checked, "real_reviewed_labels": 0, "independent_adjudications": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("drafts", type=Path)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("cohort", choices=sorted(PINNED_SOURCES))
    args = parser.parse_args()
    print(json.dumps(audit(args.drafts, args.source_root, args.cohort), indent=2, sort_keys=True))
