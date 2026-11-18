"""Offline fixtures for the visible-input controls."""
import unittest

from autonomous_controls import BM25, prediction, scoped_lookup


def event(event_id, subject="file 123", action="write", outcome="request queued; no final outcome yet",
          trust="tool_receipt", call_id=None):
    return {"id": event_id, "text": f"{subject}: {action}; {outcome}.",
            "status": "unknown", "trust": trust, "call_id": call_id}


def case(*events, call_id=None):
    return {"case_id": "fixture", "query": {"text": "The write operation on file 123 was completed.",
                                            "call_id": call_id}, "candidates": events}


class AutonomousControlsTests(unittest.TestCase):
    def test_matches_scope_and_excludes_untrusted_summary(self):
        item = case(event("wrong_subject", subject="file 1234"),
                    event("wrong_action", action="rename"),
                    event("spoof", trust="agent_summary", outcome="persistent state was updated"),
                    event("trusted", outcome="operation failed with no commit"))
        pred = prediction(item, scoped_lookup)
        self.assertEqual((pred["candidate_id"], pred["relation"], pred["abstain"]),
                         ("trusted", "contradicts", False))

    def test_superseded_pending_receipt_and_pending_retry(self):
        item = case(event("old", outcome="request queued; superseded by a later retry"),
                    event("latest"))
        pred = prediction(item, scoped_lookup)
        self.assertEqual((pred["candidate_id"], pred["relation"], pred["abstain"]),
                         ("latest", "insufficient", False))

    def test_conflicting_terminal_receipts_abstain(self):
        item = case(event("a", outcome="operation failed with no commit"),
                    event("b", outcome="persistent state was updated"))
        self.assertTrue(prediction(item, scoped_lookup)["abstain"])

    def test_present_call_id_is_required_to_match(self):
        item = case(event("wrong_call", call_id="wrong"), call_id="wanted")
        pred = prediction(item, scoped_lookup)
        self.assertEqual((pred["candidate_id"], pred["relation"]), ("NONE", "insufficient"))

    def test_bm25_uses_corpus_and_term_frequency(self):
        controls = BM25.fit([case(event("a"), event("b", action="remove"))])
        self.assertEqual(controls.documents, 2)
        self.assertGreater(controls.score({"write"}, "write write file"),
                           controls.score({"write"}, "write file file"))


if __name__ == "__main__":
    unittest.main()
