import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Karlin-Altschul lambda/K are estimated by sampling random alignments via
# biotite (see _estimate_lambda_k). These estimates are biased relative to NCBI's
# published BLOSUM62 constants (measured lambda ~0.18-0.22 vs 0.267, K ~5x low),
# so the resulting p-/E-values are an APPROXIMATE, RELATIVE-ONLY significance
# signal — useful for ranking/filtering within this tool, NOT comparable to BLAST.

# Robinson & Robinson (1991) background amino-acid frequencies, used to generate
# the random sequences sampled when estimating lambda/K.
_ROBINSON_FREQUENCIES = {
    'A': 0.0780, 'R': 0.0512, 'N': 0.0448, 'D': 0.0536, 'C': 0.0193,
    'Q': 0.0426, 'E': 0.0629, 'G': 0.0737, 'H': 0.0220, 'I': 0.0514,
    'L': 0.0901, 'K': 0.0574, 'M': 0.0224, 'F': 0.0385, 'P': 0.0520,
    'S': 0.0712, 'T': 0.0584, 'W': 0.0133, 'Y': 0.0323, 'V': 0.0644,
}


def _available_cpus() -> int:
    """Number of CPUs usable by this process.

    Prefers ``sched_getaffinity`` so it honors SLURM/cgroup CPU pinning (a job
    allocated 8 of 64 cores sees 8); falls back to ``cpu_count`` on platforms
    without it (e.g. macOS).
    """
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def _resolve_num_workers(requested: Optional[int], n_tasks: int) -> int:
    """Resolve the worker count: ``None`` -> all CPUs; never more than tasks."""
    avail = _available_cpus() if requested is None else requested
    return max(1, min(avail, n_tasks))

# The biotite ProteinSequence alphabet (20 canonical + ambiguity codes B/Z/X and
# the stop symbol *). Anything outside this set — selenocysteine U, pyrrolysine
# O, gaps, digits, whitespace — is mapped to X so alignment never crashes on a
# stray residue.
_BIOTITE_ALPHABET = set("ACDEFGHIKLMNPQRSTVWYBZX*")

# Per-process globals populated by the multiprocessing initializer so the
# (immutable, picklable-but-large) substitution matrix is built once per worker.
_MATRIX = None
_GAP: Tuple[int, int] = (-11, -1)
# Karlin-Altschul E-value estimator (biotite) and the subject-DB residue total,
# also per-process. None when significance is not being computed.
_ESTIMATOR = None
_DB_RESIDUES: Optional[int] = None


@dataclass
class AlignmentMetrics:
    """Outcome of one pairwise local alignment."""
    identity_aln: float        # identical positions / alignment length (BLAST-like pident)
    identity_shorter: float    # identical positions / length of the shorter sequence
    query_coverage: float      # aligned query residues / full query length
    subject_coverage: float    # aligned subject residues / full subject length
    aln_len: int               # alignment length (columns, gaps included)
    score: int                 # Smith-Waterman score
    # Approximate (relative-only) Karlin-Altschul significance from sampled
    # lambda/K — see module docstring; NOT calibrated against BLAST.
    pvalue: Optional[float] = None   # pairwise p-value (n = subject length)
    evalue: Optional[float] = None   # E-value over the subject DB (n = total residues)


@dataclass
class Hit:
    """A Stage-1 candidate annotated with its Stage-2 alignment metrics."""
    subject_id: str
    emb_score: float
    metrics: AlignmentMetrics


def sanitize_sequence(sequence: str) -> str:
    """Uppercase and coerce a sequence into the biotite protein alphabet."""
    return "".join(c if c in _BIOTITE_ALPHABET else "X" for c in sequence.strip().upper())


def _to_protein(sequence: str):
    """Build a biotite ProteinSequence from a (sanitized) string, or None if empty."""
    import biotite.sequence as seq
    clean = sanitize_sequence(sequence)
    if not clean:
        return None, 0
    return seq.ProteinSequence(clean), len(clean)


def _robinson_frequency_vector(alphabet) -> np.ndarray:
    """Robinson background frequencies as a vector aligned to ``alphabet`` order."""
    freq = np.array(
        [_ROBINSON_FREQUENCIES.get(sym, 0.0) for sym in alphabet.get_symbols()],
        dtype=float,
    )
    return freq / freq.sum()


def _estimate_lambda_k(
        gap_open: int,
        gap_extend: int,
        sample_size: int,
        sample_length: int,
        seed: int,
) -> Tuple[float, float]:
    """Estimate Karlin-Altschul ``(lambda, K)`` for BLOSUM62 at the given gaps
    by sampling random alignments via biotite.

    The estimate is biased relative to NCBI's published constants, so the
    derived p-/E-values are an approximate, relative-only signal (see module
    docstring). Seeded for reproducibility.
    """
    import biotite.sequence as seq
    import biotite.sequence.align as align
    alphabet = seq.ProteinSequence.alphabet
    np.random.seed(seed)
    estimator = align.EValueEstimator.from_samples(
        alphabet,
        align.SubstitutionMatrix.std_protein_matrix(),
        (-abs(gap_open), -abs(gap_extend)),
        _robinson_frequency_vector(alphabet),
        sample_length=sample_length,
        sample_size=sample_size,
    )
    logger.info(
        f"Estimated lambda={estimator.lam:.4f}, K={estimator.k:.5f} by sampling "
        f"{sample_size} alignments (approximate significance, relative-only — not BLAST-comparable)."
    )
    return float(estimator.lam), float(estimator.k)


def _evalue_from_log(log_e: float) -> float:
    """E-value from biotite's log10(E), guarding against overflow to inf."""
    with np.errstate(over='ignore'):
        return float(10.0 ** log_e)


def _pvalue_from_evalue(evalue: float) -> float:
    """p = 1 - exp(-E), numerically stable for small and large E."""
    return float(-np.expm1(-evalue))


def _worker_init(gap_open: int, gap_extend: int, lam: Optional[float], k: Optional[float],
                 db_residues: Optional[int]):
    global _MATRIX, _GAP, _ESTIMATOR, _DB_RESIDUES
    import biotite.sequence.align as align
    _MATRIX = align.SubstitutionMatrix.std_protein_matrix()  # BLOSUM62
    _GAP = (-abs(gap_open), -abs(gap_extend))
    _ESTIMATOR = align.EValueEstimator(lam, k) if lam is not None else None
    _DB_RESIDUES = db_residues


def _align(query_protein, query_len: int, subject_protein, subject_len: int) -> AlignmentMetrics:
    import biotite.sequence.align as align
    aln = align.align_optimal(
        query_protein, subject_protein, _MATRIX,
        gap_penalty=_GAP, local=True, max_number=1,
    )[0]
    score = int(aln.score)
    trace = aln.trace
    aln_len = int(trace.shape[0])
    if aln_len == 0:
        return AlignmentMetrics(0.0, 0.0, 0.0, 0.0, 0, score)
    identity_aln = float(align.get_sequence_identity(aln, mode="all"))
    identity_shorter = float(align.get_sequence_identity(aln, mode="shortest"))
    q_aligned = int((trace[:, 0] != -1).sum())
    s_aligned = int((trace[:, 1] != -1).sum())

    pvalue = evalue = None
    if _ESTIMATOR is not None:
        # Pairwise p-value: search space = the two sequences (n = subject length).
        pvalue = _pvalue_from_evalue(
            _evalue_from_log(_ESTIMATOR.log_evalue(score, query_len, subject_len))
        )
        # Database E-value: search space = full subject DB (n = total residues).
        if _DB_RESIDUES:
            evalue = _evalue_from_log(
                _ESTIMATOR.log_evalue(score, query_len, _DB_RESIDUES)
            )

    return AlignmentMetrics(
        identity_aln=identity_aln,
        identity_shorter=identity_shorter,
        query_coverage=q_aligned / query_len if query_len else 0.0,
        subject_coverage=s_aligned / subject_len if subject_len else 0.0,
        aln_len=aln_len,
        score=score,
        pvalue=pvalue,
        evalue=evalue,
    )


def _align_query(task):
    """Align one query against all of its candidate subjects. Runs in a worker."""
    query_id, query_seq, candidates = task
    query_protein, query_len = _to_protein(query_seq)
    if query_protein is None:
        return query_id, []
    hits: List[Hit] = []
    for subject_id, subject_seq, emb_score in candidates:
        subject_protein, subject_len = _to_protein(subject_seq)
        if subject_protein is None:
            continue
        metrics = _align(query_protein, query_len, subject_protein, subject_len)
        hits.append(Hit(subject_id=subject_id, emb_score=emb_score, metrics=metrics))
    return query_id, hits


def align_candidates(
        query_sequences: Dict[str, str],
        prefilter_results: Dict[str, Tuple[List[str], List[float]]],
        fetch_subject_sequences,
        min_seq_identity: float = 0.3,
        min_coverage: float = 0.0,
        gap_open: int = 11,
        gap_extend: int = 1,
        num_workers: Optional[int] = None,
        subject_db_size: Optional[int] = None,
        compute_significance: bool = True,
        # Defaults of 500x500 (vs biotite's own 1000x1000) chosen from a one-off
        # benchmark (BLOSUM62, 11/1 gaps): lambda/K had already converged to
        # within ~1% of the 1000x1000 values by 500x500, at ~8x less wall time
        # (~3s vs ~24s). Fine since the significance is relative-only anyway;
        # raise these for a steadier estimate.
        evalue_sample_size: int = 500,
        evalue_sample_length: int = 500,
        evalue_seed: int = 0,
) -> Dict[str, List[Hit]]:
    """Stage 2: pairwise-align each prefilter candidate and re-rank by identity.

    Args:
        query_sequences: ``{query_id: sequence}`` for every query.
        prefilter_results: Stage-1 output ``{query_id: ([subject_ids], [emb_scores])}``.
        fetch_subject_sequences: callable ``ids -> {id: sequence}`` (e.g.
            :meth:`SequenceStore.fetch`), used once to gather all candidate
            subject sequences across queries.
        min_seq_identity: drop hits whose ``identity_aln`` is below this.
        min_coverage: drop hits whose query *and* subject coverage are below this.
        gap_open / gap_extend: positive BLOSUM62 gap penalties (negated internally).
        num_workers: process-pool size. ``None`` (default) uses all available
            CPUs; ``0`` or ``1`` runs serially in-process. Capped at the number
            of queries.
        subject_db_size: total residue count of the subject database, used as the
            search space for the database E-value. If ``None``, no E-value is
            reported (the pairwise p-value is still computed).
        compute_significance: when ``True``, annotate each hit with an approximate
            (relative-only) Karlin-Altschul pairwise p-value and (if
            ``subject_db_size`` is given) a database E-value. These come from
            sampled lambda/K and are NOT calibrated against BLAST.
        evalue_sample_size / evalue_sample_length / evalue_seed: controls for the
            lambda/K sampling pass.

    Returns:
        ``{query_id: [Hit, ...]}`` sorted by ``identity_aln`` desc, then
        embedding score desc. Queries with no surviving hit map to ``[]``.
    """
    # Gather every candidate subject sequence in a single random-access read.
    all_subject_ids = {
        sid
        for subject_ids, _ in prefilter_results.values()
        for sid in subject_ids
    }
    subject_sequences = fetch_subject_sequences(all_subject_ids)

    tasks = []
    for query_id, (subject_ids, emb_scores) in prefilter_results.items():
        query_seq = query_sequences.get(query_id)
        if query_seq is None:
            logger.warning(f"Query '{query_id}' has no sequence; skipping alignment")
            continue
        candidates = [
            (sid, subject_sequences[sid], float(score))
            for sid, score in zip(subject_ids, emb_scores)
            if sid in subject_sequences
        ]
        missing = len(subject_ids) - len(candidates)
        if missing:
            logger.warning(f"{missing} candidate(s) for query '{query_id}' missing from sequence store; skipped")
        tasks.append((query_id, query_seq, candidates))

    # Estimate Karlin-Altschul lambda/K once (a single sampling pass) and share
    # with workers via the cheap EValueEstimator(lam, k) constructor.
    lam = k = None
    if compute_significance and tasks:
        lam, k = _estimate_lambda_k(
            gap_open, gap_extend, evalue_sample_size, evalue_sample_length, evalue_seed
        )

    n_workers = _resolve_num_workers(num_workers, len(tasks))
    init_args = (gap_open, gap_extend, lam, k, subject_db_size)
    if n_workers > 1:
        logger.info(f"Aligning {len(tasks)} queries across {n_workers} CPU workers")
        import multiprocessing as mp
        with mp.Pool(
            processes=n_workers,
            initializer=_worker_init,
            initargs=init_args,
        ) as pool:
            raw = pool.map(_align_query, tasks)
    else:
        _worker_init(*init_args)
        raw = [_align_query(task) for task in tasks]

    results: Dict[str, List[Hit]] = {}
    for query_id, hits in raw:
        kept = [
            hit for hit in hits
            if hit.metrics.identity_aln >= min_seq_identity
            and min(hit.metrics.query_coverage, hit.metrics.subject_coverage) >= min_coverage
        ]
        kept.sort(key=lambda h: (h.metrics.identity_aln, h.emb_score), reverse=True)
        results[query_id] = kept
    return results


def print_aligned_results(results: Dict[str, List[Hit]]) -> None:
    """Pretty-print Stage-2 aligned results."""
    for query_id, hits in results.items():
        logging.info(f"Query: {query_id}")
        if not hits:
            logging.info("No results found matching the criteria")
            continue
        logging.info(
            f"{'Rank':<6} {'Match':<40} {'EmbScore':<10} {'Ident':<8} "
            f"{'IdentSh':<8} {'QCov':<7} {'SCov':<7} {'AlnLen':<7} {'AlnScore':<9} "
            f"{'Pval~':<11} {'Eval~':<11}"
        )
        for rank, hit in enumerate(hits, 1):
            m = hit.metrics
            pval = f"{m.pvalue:.2e}" if m.pvalue is not None else "-"
            eval_ = f"{m.evalue:.2e}" if m.evalue is not None else "-"
            logging.info(
                f"{rank:<6} {hit.subject_id:<40} {hit.emb_score:<10.6f} "
                f"{m.identity_aln:<8.4f} {m.identity_shorter:<8.4f} "
                f"{m.query_coverage:<7.4f} {m.subject_coverage:<7.4f} "
                f"{m.aln_len:<7} {m.score:<9} {pval:<11} {eval_:<11}"
            )


def export_aligned_results(results: Dict[str, List[Hit]], output_file: str) -> None:
    """Export Stage-2 aligned results to CSV."""
    import csv
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Query', 'Rank', 'Match', 'EmbScore',
            'SeqIdentity_aln', 'SeqIdentity_shorter',
            'QueryCoverage', 'SubjectCoverage', 'AlnLen', 'AlnScore',
            'Pvalue_approx', 'Evalue_approx',
        ])
        for query_id, hits in results.items():
            for rank, hit in enumerate(hits, 1):
                m = hit.metrics
                writer.writerow([
                    query_id, rank, hit.subject_id, f"{hit.emb_score:.6f}",
                    f"{m.identity_aln:.6f}", f"{m.identity_shorter:.6f}",
                    f"{m.query_coverage:.6f}", f"{m.subject_coverage:.6f}",
                    m.aln_len, m.score,
                    f"{m.pvalue:.6e}" if m.pvalue is not None else "",
                    f"{m.evalue:.6e}" if m.evalue is not None else "",
                ])
    logging.info(f"Results exported to {output_file}")
