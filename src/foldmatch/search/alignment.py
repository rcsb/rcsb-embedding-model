import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import karlin_altschul

logger = logging.getLogger(__name__)

# Significance (p-/E-values, bit scores) comes in two flavors, selected by
# ``significance_mode``:
#
#   'default' — calibrated. Uses published ALP Karlin-Altschul constants and a
#       finite-size-corrected search space, so the values are exact and
#       reproducible (a table lookup, no sampling pass). Implementation and full
#       provenance in karlin_altschul.py. Only BLOSUM62 at gaps 11/1 is supported;
#       other gap penalties raise rather than fall back silently.
#       NOT comparable to BLAST: the nominal 11/1 gap cost here equals BLAST's
#       10/1, and these lambda/K differ from NCBI's published values.
#
#   'sampled' — the original behavior: lambda/K estimated by sampling random
#       alignments via biotite (see _estimate_lambda_k). The estimate does NOT
#       converge with sample size — measured at BLOSUM62 11/1, seed 0:
#           100x100   lambda 0.382  K 0.445
#           250x250   lambda 0.256  K 0.027
#           500x500   lambda 0.279  K 0.062   (this module's default)
#           1000x1000 lambda 0.222  K 0.008
#       i.e. K spans ~57x and lambda 0.22-0.38 purely as a function of the
#       sampling knobs. Since E scales linearly with K and exponentially in
#       lambda*S, the absolute values are unreliable; only the RANKING is
#       trustworthy (it is monotonic in the raw score for any lambda > 0).
#       Use this mode only for gap penalties 'default' cannot serve.

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


def _expand_cpu_affinity() -> int:
    """Widen this process's CPU affinity mask to all CPUs, returning the count.

    A process launched under a binding scheduler (e.g. NERSC ``srun`` pins each
    of N tasks to cpu_count/N cores) inherits a narrow mask; Stage-2 runs single-
    process on rank 0 and would otherwise see only that slice. Widening lets it
    (and its forked pool workers) use the whole node, so this must run *before*
    the pool is created.

    This is safe even on shared/allocated machines: the kernel intersects the
    requested set with the process's cpuset cgroup, so ``sched_setaffinity``
    cannot escape a real allocation — it just un-does soft ``--cpu-bind`` masks.
    No-op where ``sched_setaffinity`` is unavailable (non-Linux).
    """
    n = os.cpu_count() or 1
    if not hasattr(os, "sched_setaffinity"):
        return n
    before = len(os.sched_getaffinity(0))
    try:
        os.sched_setaffinity(0, range(n))
    except OSError as exc:
        logger.warning(f"Could not widen CPU affinity to all {n} CPUs: {exc}")
        return before
    after = len(os.sched_getaffinity(0))
    if after != before:
        logger.info(f"Widened CPU affinity for alignment: {before} -> {after} CPUs")
    return after

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
# also per-process. _ESTIMATOR is None unless significance_mode='sampled'.
_ESTIMATOR = None
_DB_RESIDUES: Optional[int] = None
# Calibrated ALP parameter set; None unless significance_mode='default'.
_ALP = None
# The active lambda/K (calibrated or sampled), kept so the bit score can be
# computed directly. None when significance is not being computed at all —
# this is the single gate for "is significance on?".
_LAMBDA: Optional[float] = None
_K: Optional[float] = None


@dataclass
class AlignmentMetrics:
    """Outcome of one pairwise local alignment."""
    identity_aln: float        # identical positions / alignment length (BLAST-like pident)
    identity_shorter: float    # identical positions / length of the shorter sequence
    query_coverage: float      # aligned query residues / full query length
    subject_coverage: float    # aligned subject residues / full subject length
    aln_len: int               # alignment length (columns, gaps included)
    score: int                 # Smith-Waterman score
    # Karlin-Altschul significance. Scale depends on significance_mode:
    # calibrated ('default') or relative-only ('sampled').
    # Never BLAST-comparable — see module header.
    bit_score: Optional[float] = None  # normalized score (lambda*S - lnK)/ln2
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


def _bit_score(score: int, lam: float, k: float) -> float:
    """Normalized (bit) score S' = (lambda*S - ln K) / ln 2.

    Uses the same lambda/K as the p-/E-values, so it inherits the active mode's
    scale: calibrated under 'default', relative-only under 'sampled'.
    Never BLAST-comparable (see module header). Because lambda/K are fixed for a
    whole run, the bit score is a strictly increasing function of the raw score,
    so ranking by bit score matches ranking by raw Smith-Waterman score in either
    mode.
    """
    return (lam * score - math.log(k)) / math.log(2.0)


def _worker_init(gap_open: int, gap_extend: int, lam: Optional[float], k: Optional[float],
                 db_residues: Optional[int], alp_params=None):
    global _MATRIX, _GAP, _ESTIMATOR, _DB_RESIDUES, _LAMBDA, _K, _ALP
    import biotite.sequence.align as align
    _MATRIX = align.SubstitutionMatrix.std_protein_matrix()  # BLOSUM62
    # Gap penalties are passed through unmodified: biotite charges a length-L gap
    # as open + (L-1)*extend, which is exactly MMseqs2's convention, so 11/1 here
    # is the scoring system its hardcoded lambda/K were shipped with.
    _GAP = (-abs(gap_open), -abs(gap_extend))
    _ALP = alp_params
    if alp_params is not None:
        _ESTIMATOR = None
        _LAMBDA, _K = alp_params.lam, alp_params.k
    else:
        _ESTIMATOR = align.EValueEstimator(lam, k) if lam is not None else None
        _LAMBDA, _K = lam, k
    _DB_RESIDUES = db_residues


def _align(query_protein, query_len: int, subject_protein, subject_len: int) -> AlignmentMetrics:
    import biotite.sequence.align as align
    aln = align.align_optimal(
        query_protein, subject_protein, _MATRIX,
        gap_penalty=_GAP, local=True, max_number=1,
    )[0]
    score = int(aln.score)
    # Compute the bit score here so both return paths carry it. None when
    # significance is off; otherwise on whatever scale the active mode implies.
    bit_score = _bit_score(score, _LAMBDA, _K) if _LAMBDA is not None else None
    trace = aln.trace
    aln_len = int(trace.shape[0])
    if aln_len == 0:
        return AlignmentMetrics(0.0, 0.0, 0.0, 0.0, 0, score, bit_score=bit_score)
    identity_aln = float(align.get_sequence_identity(aln, mode="all"))
    identity_shorter = float(align.get_sequence_identity(aln, mode="shortest"))
    q_aligned = int((trace[:, 0] != -1).sum())
    s_aligned = int((trace[:, 1] != -1).sum())

    pvalue = evalue = None
    if _ALP is not None:
        # Calibrated: ALP constants + ALP's finite-size-corrected area.
        # Pairwise p-value: search space = the two sequences (n = subject length).
        # MMseqs2 itself emits no p-value; this is the ALP p-value of the
        # pairwise E, kept for continuity with the 'sampled' mode.
        pvalue = karlin_altschul.pvalue_from_evalue(
            karlin_altschul.evalue(score, query_len, subject_len, _ALP)
        )
        # Database E-value: this is the MMseqs2-comparable quantity — search
        # space = the whole target DB as one concatenated sequence, exactly as
        # MMseqs2 does (no per-sequence term, no multiply by sequence count).
        if _DB_RESIDUES:
            evalue = karlin_altschul.evalue(score, query_len, _DB_RESIDUES, _ALP)
    elif _ESTIMATOR is not None:
        # Sampled/relative-only: biotite's raw K*m*n*exp(-lambda*S), no edge correction.
        pvalue = _pvalue_from_evalue(
            _evalue_from_log(_ESTIMATOR.log_evalue(score, query_len, subject_len))
        )
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
        bit_score=bit_score,
        pvalue=pvalue,
        evalue=evalue,
    )


def _align_query(task):
    """Align one query against a chunk of its candidate subjects. Runs in a worker.

    The chunk may be all of a query's candidates or a slice of them (see
    :func:`_chunk_candidate_tasks`); either way the query protein is built once
    for the chunk and a partial hit list is returned, keyed by ``query_id`` so
    the caller can regroup a query that was split across several chunks.
    """
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


def _chunk_candidate_tasks(tasks, worker_budget: int, tasks_per_worker: int = 4):
    """Split per-query tasks into per-(query, candidate-chunk) tasks for the pool.

    Parallelizing over whole queries pins a single query with many candidates to
    one worker (every other core idle); splitting a query's candidate list into
    chunks lets the pool spread those alignments across all workers. Chunk size
    is chosen so there are roughly ``tasks_per_worker * worker_budget`` chunks in
    total — enough to keep every worker fed and to smooth out the load imbalance
    from variable-length (quadratic-cost) alignments — while still batching
    several candidates per chunk to amortize the per-chunk query rebuild and the
    task's pickling overhead.

    Queries with no candidates are dropped here (they emit no work); the caller
    re-seeds them so they still report an empty hit list.
    """
    total_pairs = sum(len(candidates) for _, _, candidates in tasks)
    if total_pairs == 0:
        return list(tasks)
    target_chunks = tasks_per_worker * max(1, worker_budget)
    chunk_size = max(1, math.ceil(total_pairs / target_chunks))
    chunked = []
    for query_id, query_seq, candidates in tasks:
        for start in range(0, len(candidates), chunk_size):
            chunked.append((query_id, query_seq, candidates[start:start + chunk_size]))
    return chunked


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
        significance_mode: str = "default",
        # 'sampled' mode only. NOTE: an earlier comment here claimed lambda/K had
        # converged to within ~1% of the 1000x1000 values by 500x500 — that is
        # measurably wrong (lambda 0.279 vs 0.222, K 0.062 vs 0.008; see the
        # table in the module header). Raising these does NOT buy a steadier
        # estimate, so treat 'sampled' absolute values as unreliable and prefer
        # significance_mode='default' wherever the gap penalties allow it.
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
        num_workers: process-pool size. ``None`` (default) uses **all** CPUs:
            it first widens the process CPU-affinity mask (undoing any scheduler
            ``--cpu-bind`` pinning, kernel-clamped to the cgroup allocation) so a
            binding launcher like ``srun`` can't cap it below the node's cores.
            ``0`` or ``1`` runs serially in-process. An explicit ``N > 1`` uses
            N workers (also widening affinity so they can spread). Capped at the
            number of queries.
        subject_db_size: total residue count of the subject database, used as the
            search space for the database E-value. If ``None``, no E-value is
            reported (the pairwise p-value is still computed).
        compute_significance: when ``True``, annotate each hit with a bit score,
            a pairwise p-value and (if ``subject_db_size`` is given) a database
            E-value.
        significance_mode: ``'default'`` uses calibrated Karlin-Altschul
            constants with a finite-size-corrected search space, giving exact,
            reproducible values; it supports only ``gap_open=11, gap_extend=1``
            and raises otherwise. ``'sampled'`` restores the previous behavior —
            lambda/K estimated by sampling, whose absolute values are unreliable
            (they do not converge with sample size) though the ranking is sound;
            use it for gap penalties ``'default'`` cannot serve. Neither mode is
            BLAST-comparable; see the module header.
        evalue_sample_size / evalue_sample_length / evalue_seed: controls for the
            lambda/K sampling pass (``'sampled'`` mode only).

    Returns:
        ``{query_id: [Hit, ...]}`` sorted by ``bit_score`` desc (equivalently
        raw alignment score, since lambda/K are shared across the run), then
        embedding score desc; with ``compute_significance=False`` the raw
        alignment score is used directly. Queries with no surviving hit map to
        ``[]``.
    """
    start_time = time.perf_counter()
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

    # Resolve the significance parameters once and share them with the workers.
    # 'default' is a table lookup (no sampling pass); 'sampled' runs the estimator.
    lam = k = alp_params = None
    if compute_significance and tasks:
        if significance_mode == "default":
            alp_params = karlin_altschul.params_for(gap_open, gap_extend)
            logger.info(
                f"Using calibrated significance (BLOSUM62 {gap_open}/{gap_extend}: "
                f"lambda={alp_params.lam:.8f}, K={alp_params.k:.9f}); "
                f"E-values are not comparable to BLAST."
            )
        elif significance_mode == "sampled":
            lam, k = _estimate_lambda_k(
                gap_open, gap_extend, evalue_sample_size, evalue_sample_length, evalue_seed
            )
        else:
            raise ValueError(
                f"Unknown significance_mode {significance_mode!r}; "
                f"expected 'default' or 'sampled'."
            )

    # Unless explicitly serial, widen the affinity mask before sizing the pool
    # so a scheduler-pinned rank (e.g. srun-bound) still uses the whole node and
    # the forked workers inherit the widened mask. Kernel-clamped to the cgroup,
    # so this can't grab cores outside a real allocation.
    serial = num_workers is not None and num_workers <= 1
    if not serial and tasks:
        _expand_cpu_affinity()

    init_args = (gap_open, gap_extend, lam, k, subject_db_size, alp_params)
    if not serial and tasks:
        # Parallelize over (query, subject) *pairs* rather than whole queries by
        # chunking each query's candidate list, so even a single query with many
        # candidates uses the whole node instead of one core. Size the pool by
        # the number of chunks (capped, as before, at the worker budget).
        worker_budget = _available_cpus() if num_workers is None else num_workers
        pair_tasks = _chunk_candidate_tasks(tasks, worker_budget)
        n_workers = _resolve_num_workers(num_workers, len(pair_tasks))
    else:
        pair_tasks = tasks
        n_workers = 1

    if n_workers > 1:
        total_pairs = sum(len(candidates) for _, _, candidates in tasks)
        logger.info(
            f"Aligning {total_pairs} (query, subject) pairs from {len(tasks)} "
            f"queries in {len(pair_tasks)} chunks across {n_workers} CPU workers"
        )
        import multiprocessing as mp
        with mp.Pool(
            processes=n_workers,
            initializer=_worker_init,
            initargs=init_args,
        ) as pool:
            # chunksize=1: each candidate-chunk is its own dispatch unit (we've
            # already sized the chunks) so variable-length alignments stay
            # balanced; pool.map preserves task order.
            raw = pool.map(_align_query, pair_tasks, chunksize=1)
    else:
        _worker_init(*init_args)
        raw = [_align_query(task) for task in pair_tasks]

    # A query may now span several chunks, so accumulate its hits before
    # filtering/sorting. Pre-seed every query (in prefilter order) so those with
    # no candidates still map to [] and the result ordering is preserved.
    per_query: Dict[str, List[Hit]] = {query_id: [] for query_id, _, _ in tasks}
    for query_id, hits in raw:
        per_query[query_id].extend(hits)

    results: Dict[str, List[Hit]] = {}
    for query_id, hits in per_query.items():
        kept = [
            hit for hit in hits
            if hit.metrics.identity_aln >= min_seq_identity
            and min(hit.metrics.query_coverage, hit.metrics.subject_coverage) >= min_coverage
        ]
        # Rank by bit score (desc), embedding score breaking ties. When
        # significance is disabled there is no bit score, so fall back to the raw
        # alignment score — the same ordering, since the bit score is monotonic
        # in the raw score under the run-wide shared lambda/K.
        kept.sort(
            key=lambda h: (
                h.metrics.bit_score if h.metrics.bit_score is not None else h.metrics.score,
                h.emb_score,
            ),
            reverse=True,
        )
        results[query_id] = kept
    logger.info(f"Pairwise alignments completed in {time.perf_counter() - start_time:.2f}s")
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
            f"{'Bits':<9} {'Pval':<11} {'Eval':<11}"
        )
        for rank, hit in enumerate(hits, 1):
            m = hit.metrics
            bits = f"{m.bit_score:.1f}" if m.bit_score is not None else "-"
            pval = f"{m.pvalue:.2e}" if m.pvalue is not None else "-"
            eval_ = f"{m.evalue:.2e}" if m.evalue is not None else "-"
            logging.info(
                f"{rank:<6} {hit.subject_id:<40} {hit.emb_score:<10.6f} "
                f"{m.identity_aln:<8.4f} {m.identity_shorter:<8.4f} "
                f"{m.query_coverage:<7.4f} {m.subject_coverage:<7.4f} "
                f"{m.aln_len:<7} {m.score:<9} {bits:<9} {pval:<11} {eval_:<11}"
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
            'BitScore', 'Pvalue', 'Evalue',
        ])
        for query_id, hits in results.items():
            for rank, hit in enumerate(hits, 1):
                m = hit.metrics
                writer.writerow([
                    query_id, rank, hit.subject_id, f"{hit.emb_score:.6f}",
                    f"{m.identity_aln:.6f}", f"{m.identity_shorter:.6f}",
                    f"{m.query_coverage:.6f}", f"{m.subject_coverage:.6f}",
                    m.aln_len, m.score,
                    f"{m.bit_score:.4f}" if m.bit_score is not None else "",
                    f"{m.pvalue:.6e}" if m.pvalue is not None else "",
                    f"{m.evalue:.6e}" if m.evalue is not None else "",
                ])
    logging.info(f"Results exported to {output_file}")
