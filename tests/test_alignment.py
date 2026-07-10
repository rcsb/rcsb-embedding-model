import unittest

from foldmatch.search import alignment
from foldmatch.search.alignment import _chunk_candidate_tasks


# A small, fixed set of short protein sequences. Absolute identity values don't
# matter for these tests — only that serial and parallel paths agree and that
# every candidate is accounted for — so any deterministic sequences will do.
_STORE = {
    "s1": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ",
    "s2": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVA",   # 1 substitution vs s1
    "s3": "MKTAYIAKQRQISFVKSHFSRQ",              # prefix of s1
    "s4": "GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG",   # unrelated
    "s5": "PLYISNDACEFHIKLMNPQRSTVWYACDEFGHIK",
    "s6": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEEE",
    "s7": "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQ",
    "s8": "MKTAYIAKQRQISFVKSHFSRQLEERLG",
}


def _fetch(ids):
    return {i: _STORE[i] for i in ids if i in _STORE}


def _signature(results):
    """A hashable, order-sensitive view of the aligned results for comparison."""
    out = {}
    for query_id, hits in results.items():
        out[query_id] = [
            (
                h.subject_id,
                round(h.metrics.identity_aln, 6),
                round(h.metrics.query_coverage, 6),
                round(h.metrics.subject_coverage, 6),
                h.metrics.score,
            )
            for h in hits
        ]
    return out


class TestChunkCandidateTasks(unittest.TestCase):
    def test_preserves_all_pairs_in_order(self):
        q1 = [(f"a{i}", "SEQ", float(i)) for i in range(10)]
        q2 = [(f"b{i}", "SEQ", float(i)) for i in range(3)]
        tasks = [("q1", "AAAA", q1), ("q2", "BBBB", q2)]

        chunks = _chunk_candidate_tasks(tasks, worker_budget=2)  # -> chunk_size 2

        # No chunk mixes queries and none exceeds the computed chunk size.
        self.assertTrue(all(len(c) <= 2 for _, _, c in chunks))
        # Concatenating each query's chunks reproduces its candidate list in order.
        regrouped = {}
        for qid, _, cand in chunks:
            regrouped.setdefault(qid, []).extend(cand)
        self.assertEqual(regrouped["q1"], q1)
        self.assertEqual(regrouped["q2"], q2)

    def test_single_query_is_split_across_many_chunks(self):
        cand = [(f"s{i}", "SEQ", float(i)) for i in range(64)]
        tasks = [("q", "AAAA", cand)]
        chunks = _chunk_candidate_tasks(tasks, worker_budget=8)
        # A lone query must fan out to more than one chunk so the pool can spread
        # it — this is the whole point of the change.
        self.assertGreater(len(chunks), 1)

    def test_empty_and_no_pairs(self):
        self.assertEqual(_chunk_candidate_tasks([], worker_budget=4), [])
        tasks = [("q", "AAAA", [])]
        # No pairs to align: returned as-is (caller re-seeds it to []).
        self.assertEqual(_chunk_candidate_tasks(tasks, worker_budget=4), tasks)


class TestAlignCandidatesParallelism(unittest.TestCase):
    def _run(self, prefilter, queries, num_workers):
        return alignment.align_candidates(
            query_sequences=queries,
            prefilter_results=prefilter,
            fetch_subject_sequences=_fetch,
            min_seq_identity=0.0,   # keep every hit so completeness is checkable
            min_coverage=0.0,
            num_workers=num_workers,
            compute_significance=False,  # skip the slow lambda/K sampling pass
        )

    def test_serial_and_parallel_agree_single_query(self):
        # One query with many candidates: the case that previously pinned to one
        # core. Serial (workers=1) and pooled (workers=2, forces chunking) must
        # return byte-for-byte identical metrics and ordering.
        queries = {"q1": _STORE["s1"]}
        sids = [f"s{i}" for i in range(1, 9)]
        prefilter = {"q1": (sids, [1.0 / i for i in range(1, 9)])}

        serial = self._run(prefilter, queries, num_workers=1)
        parallel = self._run(prefilter, queries, num_workers=2)

        self.assertEqual(_signature(serial), _signature(parallel))
        # All eight candidates survive the (disabled) filters.
        self.assertEqual(len(parallel["q1"]), 8)

    def test_serial_and_parallel_agree_multi_query(self):
        queries = {"q1": _STORE["s1"], "q2": _STORE["s5"], "q3": _STORE["s7"]}
        prefilter = {
            "q1": (["s1", "s2", "s3", "s6", "s8"], [0.9, 0.8, 0.7, 0.6, 0.5]),
            "q2": (["s5", "s7", "s4"], [0.95, 0.55, 0.2]),
            "q3": (["s7", "s5"], [0.99, 0.4]),
        }
        serial = self._run(prefilter, queries, num_workers=1)
        parallel = self._run(prefilter, queries, num_workers=4)
        self.assertEqual(_signature(serial), _signature(parallel))
        # Result ordering (query order) is preserved.
        self.assertEqual(list(serial.keys()), list(parallel.keys()))

    def test_query_with_no_candidates_maps_to_empty(self):
        queries = {"q1": _STORE["s1"], "q2": _STORE["s5"]}
        prefilter = {
            "q1": (["s1", "s2"], [0.9, 0.8]),
            "q2": ([], []),          # nothing survived the prefilter
        }
        res = self._run(prefilter, queries, num_workers=2)
        self.assertIn("q2", res)
        self.assertEqual(res["q2"], [])
        self.assertEqual(len(res["q1"]), 2)


if __name__ == "__main__":
    unittest.main()
