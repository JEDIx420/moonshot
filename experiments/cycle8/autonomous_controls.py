#!/usr/bin/env python3
"""Visible-input Cycle 8B validation controls. No oracle files opened here."""

import argparse
from collections import Counter
from dataclasses import dataclass
import math
from pathlib import Path
import re

from autonomous_oracle import jlines, polarity, save_jsonl, text_status, TOKEN

CLAIM = re.compile(r"The (.+?) operation on (.+?) (?:was (?:completed|rejected)|failed|went through successfully)\.", re.I)
STOP = {"the", "on", "operation", "was", "completed", "failed",
        "went", "through", "successfully", "rejected"}


def terms(text):
    return TOKEN.findall(text.lower())


def receipt_matches(event, subject, action):
    observed_subject, colon, rest = event["text"].partition(":")
    observed_action, semicolon, _ = rest.partition(";")
    return bool(colon and semicolon and terms(observed_subject) == terms(subject)
                and terms(observed_action) == terms(action))


def trusted_receipts(case):
    return [event for event in case["candidates"] if event["trust"] == "tool_receipt"]


def scoped_lookup(case):
    query = case["query"]
    match = CLAIM.fullmatch(query["text"].strip())
    if not match:
        return None, True
    action, subject = match.groups()
    receipts = trusted_receipts(case)
    if query["call_id"]:
        receipts = [event for event in receipts if event["call_id"] == query["call_id"]]
    matches = [event for event in receipts if receipt_matches(event, subject, action)]
    # A receipt explicitly marked as superseded cannot establish the later retry.
    # Use that cue only when a surviving receipt for the same operation exists.
    unsuperseded = [event for event in matches
                    if "superseded by a later retry" not in event["text"].lower()]
    if unsuperseded:
        matches = unsuperseded
    terminal = [event for event in matches if text_status(event) in ("committed", "failed")]
    if len(terminal) == 1:
        return terminal[0], False
    if len(terminal) > 1 or len(matches) > 1:
        return None, True
    return (matches[0] if matches else None), False


@dataclass
class BM25:
    documents: int
    average_length: float
    frequency: Counter

    @classmethod
    def fit(cls, cases):
        seen = set()
        corpus = []
        for case in cases:
            receipts = trusted_receipts(case)
            # Opposing claims have the same receipts. Count each receipt set once.
            key = tuple(sorted(event["text"] for event in receipts))
            if key not in seen:
                seen.add(key)
                corpus.extend(terms(text) for text in key)
        if not corpus:
            raise ValueError("empty BM25 training corpus")
        return cls(len(corpus), sum(map(len, corpus)) / len(corpus),
                   Counter(token for doc in corpus for token in set(doc)))

    def score(self, query, document):
        words = terms(document)
        counts = Counter(words)
        norm = 1.2 * (0.25 + 0.75 * len(words) / self.average_length)
        score = 0.0
        for term in sorted(query & counts.keys()):
            frequency = self.frequency[term]
            idf = math.log1p((self.documents - frequency + 0.5) / (frequency + 0.5))
            score += idf * (2.2 * counts[term] / (counts[term] + norm))
        return score

    def lookup(self, case):
        query = set(terms(case["query"]["text"])) - STOP
        ranked = sorted(((self.score(query, event["text"]), event["id"], event)
                         for event in trusted_receipts(case)), key=lambda row: (-row[0], row[1]))
        if not ranked or ranked[0][0] <= 0 or (len(ranked) > 1 and ranked[0][0] == ranked[1][0]):
            return None, True
        return ranked[0][2], False


def prediction(case, lookup):
    event, abstain = lookup(case)
    claimed = polarity(case["query"]["text"])
    observed = text_status(event) if event else None
    relation = "insufficient"
    if not abstain and claimed and observed in ("committed", "failed"):
        relation = "supports" if claimed == observed else "contradicts"
    return {"case_id": case["case_id"], "candidate_id": event["id"] if event else "NONE",
            "relation": relation, "abstain": abstain or claimed is None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("scoped_receipt", "corpus_bm25"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source = args.dataset_dir / "inputs"
    lookup = (scoped_lookup if args.mode == "scoped_receipt"
              else BM25.fit(jlines(source / "train.jsonl")).lookup)
    save_jsonl(args.output, (prediction(case, lookup) for case in jlines(source / "validation.jsonl")))


if __name__ == "__main__":
    main()
