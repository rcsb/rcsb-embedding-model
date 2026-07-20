import re
import tempfile
import unittest
from pathlib import Path

from foldmatch.search import alignment
from foldmatch.search.alignment import (
    DEFAULT_OUTPUT_FIELDS,
    SUPPORTED_OUTPUT_FIELDS,
    _chunk_candidate_tasks,
    parse_format_output,
)


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


class TestParseFormatOutput(unittest.TestCase):
    def test_default_when_none(self):
        self.assertEqual(parse_format_output(None), list(DEFAULT_OUTPUT_FIELDS))

    def test_preserves_order_and_repeats(self):
        self.assertEqual(
            parse_format_output("target,query,target"),
            ["target", "query", "target"],
        )

    def test_strips_whitespace_and_blanks(self):
        self.assertEqual(parse_format_output(" query , target ,"), ["query", "target"])

    def test_unknown_field_raises_with_supported_list(self):
        with self.assertRaises(ValueError) as ctx:
            parse_format_output("query,bogus,target")
        msg = str(ctx.exception)
        self.assertIn("bogus", msg)
        # The error advertises the full supported set so the user can self-correct.
        self.assertIn("qaln", msg)

    def test_empty_spec_raises(self):
        with self.assertRaises(ValueError):
            parse_format_output("  ,  ")

    def test_every_supported_field_is_renderable(self):
        # Registry and renderer table must stay in lock-step.
        self.assertEqual(
            set(SUPPORTED_OUTPUT_FIELDS),
            set(alignment._FIELD_RENDERERS),
        )


def _cigar_counts(cigar):
    """Sum of each op's run lengths in a CIGAR string, e.g. '7M1D3I' -> {M:7,D:1,I:3}."""
    counts = {"M": 0, "D": 0, "I": 0}
    for num, op in re.findall(r"(\d+)([MDI])", cigar):
        counts[op] += int(num)
    return counts


def _gap_runs(gapped):
    """Number of maximal '-' runs in an aligned string."""
    return len(re.findall(r"-+", gapped))


class TestOutputFieldComputation(unittest.TestCase):
    """The mmseqs-style columns must satisfy their defining invariants."""

    def _all_fields_hits(self, num_workers):
        queries = {"q1": _STORE["s1"], "q2": _STORE["s5"]}
        prefilter = {
            "q1": (["s1", "s2", "s3", "s6", "s8"], [0.9, 0.8, 0.7, 0.6, 0.5]),
            "q2": (["s5", "s7", "s4"], [0.95, 0.55, 0.2]),
        }
        return alignment.align_candidates(
            query_sequences=queries,
            prefilter_results=prefilter,
            fetch_subject_sequences=_fetch,
            min_seq_identity=0.0,
            min_coverage=0.0,
            num_workers=num_workers,
            subject_db_size=sum(len(v) for v in _STORE.values()),
            compute_significance=False,
            output_fields=list(SUPPORTED_OUTPUT_FIELDS),
        )

    def test_field_invariants(self):
        results = self._all_fields_hits(num_workers=1)
        seen = 0
        for query_id, hits in results.items():
            for hit in hits:
                seen += 1
                m = hit.metrics
                # Sequences reported are exactly what was aligned.
                self.assertIsNotNone(m.q_seq)
                self.assertIsNotNone(m.t_seq)
                self.assertEqual(m.query_len, len(m.q_seq))
                self.assertEqual(m.subject_len, len(m.t_seq))

                # Aligned strings line up with the alignment length.
                self.assertEqual(len(m.q_aln), m.aln_len)
                self.assertEqual(len(m.t_aln), m.aln_len)

                # Column bookkeeping: alnlen = matches + mismatches + gap columns.
                q_gaps = m.q_aln.count("-")
                t_gaps = m.t_aln.count("-")
                self.assertEqual(m.aln_len, m.n_ident + m.mismatch + q_gaps + t_gaps)

                # fident is nident/alnlen.
                self.assertAlmostEqual(m.identity_aln, m.n_ident / m.aln_len, places=9)

                # CIGAR is consistent with the aligned strings.
                cc = _cigar_counts(m.cigar)
                self.assertEqual(sum(cc.values()), m.aln_len)
                self.assertEqual(cc["M"], m.n_ident + m.mismatch)
                self.assertEqual(cc["D"], q_gaps)   # D = gap in query
                self.assertEqual(cc["I"], t_gaps)   # I = gap in target

                # gapopen counts runs, not characters.
                self.assertEqual(m.gap_open, _gap_runs(m.q_aln) + _gap_runs(m.t_aln))

                # 1-indexed bounds within the sequences, and coverage matches them.
                self.assertTrue(1 <= m.q_start <= m.q_end <= m.query_len)
                self.assertTrue(1 <= m.t_start <= m.t_end <= m.subject_len)
                self.assertEqual(m.q_end - m.q_start + 1, m.aln_len - q_gaps)
                self.assertEqual(m.t_end - m.t_start + 1, m.aln_len - t_gaps)
                self.assertAlmostEqual(
                    m.query_coverage, (m.aln_len - q_gaps) / m.query_len, places=9
                )
                self.assertAlmostEqual(
                    m.subject_coverage, (m.aln_len - t_gaps) / m.subject_len, places=9
                )

                # The non-gap columns of qaln/taln are substrings of the sequences.
                self.assertEqual(m.q_aln.replace("-", ""), m.q_seq[m.q_start - 1:m.q_end])
                self.assertEqual(m.t_aln.replace("-", ""), m.t_seq[m.t_start - 1:m.t_end])
        self.assertGreater(seen, 0)

    def test_serial_and_parallel_agree_on_all_fields(self):
        def sig(results):
            return {
                qid: [alignment.format_row(qid, h, list(SUPPORTED_OUTPUT_FIELDS), "|") for h in hits]
                for qid, hits in results.items()
            }
        self.assertEqual(
            sig(self._all_fields_hits(1)),
            sig(self._all_fields_hits(2)),
        )


class TestDefaultFormatGating(unittest.TestCase):
    def test_heavy_fields_not_materialized_by_default(self):
        res = alignment.align_candidates(
            query_sequences={"q1": _STORE["s1"]},
            prefilter_results={"q1": (["s1", "s2"], [0.9, 0.8])},
            fetch_subject_sequences=_fetch,
            min_seq_identity=0.0,
            num_workers=1,
            subject_db_size=sum(len(v) for v in _STORE.values()),
            compute_significance=False,
            # default output_fields (None) -> no heavy strings requested
        )
        m = res["q1"][0].metrics
        for attr in ("cigar", "q_aln", "t_aln", "q_seq", "t_seq"):
            self.assertIsNone(getattr(m, attr), f"{attr} should be gated off by default")
        # Scalars are always available regardless of format.
        self.assertGreater(m.n_ident, 0)
        self.assertEqual(m.q_start, 1)


class TestNoAlignmentCandidateDropped(unittest.TestCase):
    def test_empty_alignment_is_not_reported_even_at_zero_thresholds(self):
        # Two sequences that share no positive-scoring local alignment produce an
        # empty biotite trace (aln_len == 0). Such a candidate is not a hit and
        # must be dropped rather than emitted as a malformed position-0 row —
        # even with the identity/coverage thresholds relaxed all the way to 0.
        store = {"far": "PPPPPPPPPP"}
        res = alignment.align_candidates(
            query_sequences={"q": "WWWWWWWWWW"},
            prefilter_results={"q": (["far"], [0.99])},
            fetch_subject_sequences=lambda ids: {i: store[i] for i in ids if i in store},
            min_seq_identity=0.0,
            min_coverage=0.0,
            num_workers=1,
            subject_db_size=10,
            output_fields=list(SUPPORTED_OUTPUT_FIELDS),
        )
        self.assertEqual(res["q"], [])


class TestWriteAlignedResults(unittest.TestCase):
    def test_tsv_no_header(self):
        fields = ["query", "target", "fident", "alnlen", "cigar"]
        res = alignment.align_candidates(
            query_sequences={"q1": _STORE["s1"]},
            prefilter_results={"q1": (["s1", "s2"], [0.9, 0.8])},
            fetch_subject_sequences=_fetch,
            min_seq_identity=0.0,
            num_workers=1,
            subject_db_size=sum(len(v) for v in _STORE.values()),
            compute_significance=False,
            output_fields=fields,
        )
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "hits.tsv"
            alignment.write_aligned_results(res, fields, str(out))
            lines = out.read_text().splitlines()

        self.assertEqual(len(lines), 2)  # two hits, no header row
        for line in lines:
            cols = line.split("\t")
            self.assertEqual(len(cols), len(fields))
            self.assertEqual(cols[0], "q1")     # query id
        # First hit is the identical self-match: fident 1.000, cigar full-length M.
        self.assertEqual(lines[0].split("\t")[2], "1.000")
        self.assertRegex(lines[0].split("\t")[4], r"^\d+M$")


if __name__ == "__main__":
    unittest.main()
