import math
import unittest

from foldmatch.search import karlin_altschul as ms

P = ms.BLOSUM62_GAPPED_11_1

# Golden values captured from a real `mmseqs easy-search` run (mmseqs 18-8cc5c,
# BLOSUM62, --gap-open 11 --gap-extend 1 --comp-bias-corr 0 --mask 0) against a
# 60-sequence target database totalling 18850 residues.
#
# Note mmseqs's own `raw` output column is reconstructed from its stored INTEGER
# bit score and is lossy by +/-1; the scores below are the true integer scores,
# recovered as the unique value consistent with BOTH the reported bit score and
# the reported E-value. mmseqs prints E-values to 4 significant digits, which
# sets the comparison tolerance.
DB_RESIDUES = 18850
GOLDEN = [  # (score, query_len, mmseqs_bits, mmseqs_evalue)
    (2400, 596, 952, 5.544e-314),
    (759, 606, 304, 9.127e-91),
    (320, 600, 131, 9.03e-34),
    (47, 280, 23, 0.4912),
    (42, 596, 21, 4.708),
    (39, 598, 20, 10.83),
    (37, 602, 19, 18.97),
    (34, 598, 18, 43.17),
    (31, 278, 17, 43.26),
]


class TestAgainstRealMmseqs(unittest.TestCase):
    def test_evalues_match_mmseqs(self):
        for score, qlen, _bits, expected in GOLDEN:
            got = ms.evalue(score, qlen, DB_RESIDUES, P)
            rel = abs(got - expected) / expected
            self.assertLess(
                rel, 1e-3,
                f"score={score} qlen={qlen}: got {got:.6e}, mmseqs {expected:.6e} "
                f"(rel {rel:.2e})",
            )

    def test_bit_scores_match_mmseqs(self):
        # mmseqs emits int(bitScore + 0.5).
        for score, _qlen, expected_bits, _e in GOLDEN:
            self.assertEqual(int(ms.bit_score(score, P) + 0.5), expected_bits)


class TestAlpInternals(unittest.TestCase):
    def test_half_term_matches_closed_form(self):
        # p should equal li*Phi(F) + sqrt_v*phi(F); guards the sign convention,
        # which is written in ALP as a minus against a negative phi.
        for length, y in [(100, 50), (300, 120), (1e7, 200), (1000, 400)]:
            got, _ = ms._half_term(length, P.a_i, P.b_i, P.alpha_i, P.beta_i,
                                   P.vi_y_thr, y)
            li = length - (P.a_i * y + P.b_i)
            v = max(P.vi_y_thr, P.alpha_i * y + P.beta_i)
            s = math.sqrt(v)
            f = li / s
            phi = math.exp(-0.5 * f * f) / math.sqrt(2 * math.pi)
            expected = li * 0.5 * math.erfc(-math.sqrt(0.5) * f) + s * phi
            self.assertAlmostEqual(got, expected, delta=1e-9 * max(1.0, abs(expected)))

    def test_evalue_decreases_with_score(self):
        prev = None
        for s in [30, 50, 80, 120, 200, 400]:
            e = ms.evalue(s, 350, 10 ** 8, P)
            if prev is not None:
                self.assertLess(e, prev)
            prev = e

    def test_pvalue_piecewise_agrees_with_expm1(self):
        # ALP uses a Taylor form below |y|=1e-3; it must still be accurate.
        for e in [1e-9, 5e-4, 1e-2, 5.0]:
            self.assertAlmostEqual(ms.pvalue_from_evalue(e), -math.expm1(-e),
                                   delta=1e-12)

    def test_rejects_unsupported_gap_penalties(self):
        # MMseqs2 hardcodes only 11/1 for BLOSUM62; anything else must refuse
        # rather than silently substituting different constants.
        ms.params_for(11, 1)  # supported
        for gaps in [(13, 1), (10, 1), (11, 2), (0, 0)]:
            with self.assertRaises(ValueError):
                ms.params_for(*gaps)

    def test_rejects_nonpositive_lengths(self):
        for lens in [(0, 100), (100, 0), (-1, 100)]:
            with self.assertRaises(ValueError):
                ms.area(100, lens[0], lens[1], P)


if __name__ == "__main__":
    unittest.main()
