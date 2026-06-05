import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# The biotite ProteinSequence alphabet (20 canonical + ambiguity codes B/Z/X and
# the stop symbol *). Anything outside this set — selenocysteine U, pyrrolysine
# O, gaps, digits, whitespace — is mapped to X so alignment never crashes on a
# stray residue.
_BIOTITE_ALPHABET = set("ACDEFGHIKLMNPQRSTVWYBZX*")

# Per-process globals populated by the multiprocessing initializer so the
# (immutable, picklable-but-large) substitution matrix is built once per worker.
_MATRIX = None
_GAP: Tuple[int, int] = (-11, -1)


@dataclass
class AlignmentMetrics:
    """Outcome of one pairwise local alignment."""
    identity_aln: float        # identical positions / alignment length (BLAST-like pident)
    identity_shorter: float    # identical positions / length of the shorter sequence
    query_coverage: float      # aligned query residues / full query length
    subject_coverage: float    # aligned subject residues / full subject length
    aln_len: int               # alignment length (columns, gaps included)
    score: int                 # Smith-Waterman score


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


def _worker_init(gap_open: int, gap_extend: int):
    global _MATRIX, _GAP
    import biotite.sequence.align as align
    _MATRIX = align.SubstitutionMatrix.std_protein_matrix()  # BLOSUM62
    _GAP = (-abs(gap_open), -abs(gap_extend))


def _align(query_protein, query_len: int, subject_protein, subject_len: int) -> AlignmentMetrics:
    import biotite.sequence.align as align
    aln = align.align_optimal(
        query_protein, subject_protein, _MATRIX,
        gap_penalty=_GAP, local=True, max_number=1,
    )[0]
    trace = aln.trace
    aln_len = int(trace.shape[0])
    if aln_len == 0:
        return AlignmentMetrics(0.0, 0.0, 0.0, 0.0, 0, int(aln.score))
    identity_aln = float(align.get_sequence_identity(aln, mode="all"))
    identity_shorter = float(align.get_sequence_identity(aln, mode="shortest"))
    q_aligned = int((trace[:, 0] != -1).sum())
    s_aligned = int((trace[:, 1] != -1).sum())
    return AlignmentMetrics(
        identity_aln=identity_aln,
        identity_shorter=identity_shorter,
        query_coverage=q_aligned / query_len if query_len else 0.0,
        subject_coverage=s_aligned / subject_len if subject_len else 0.0,
        aln_len=aln_len,
        score=int(aln.score),
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
        num_workers: int = 0,
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
        num_workers: process-pool size; <=1 runs serially in-process.

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

    if num_workers and num_workers > 1:
        import multiprocessing as mp
        with mp.Pool(
            processes=num_workers,
            initializer=_worker_init,
            initargs=(gap_open, gap_extend),
        ) as pool:
            raw = pool.map(_align_query, tasks)
    else:
        _worker_init(gap_open, gap_extend)
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
            f"{'IdentSh':<8} {'QCov':<7} {'SCov':<7} {'AlnLen':<7} {'AlnScore':<9}"
        )
        for rank, hit in enumerate(hits, 1):
            m = hit.metrics
            logging.info(
                f"{rank:<6} {hit.subject_id:<40} {hit.emb_score:<10.6f} "
                f"{m.identity_aln:<8.4f} {m.identity_shorter:<8.4f} "
                f"{m.query_coverage:<7.4f} {m.subject_coverage:<7.4f} "
                f"{m.aln_len:<7} {m.score:<9}"
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
        ])
        for query_id, hits in results.items():
            for rank, hit in enumerate(hits, 1):
                m = hit.metrics
                writer.writerow([
                    query_id, rank, hit.subject_id, f"{hit.emb_score:.6f}",
                    f"{m.identity_aln:.6f}", f"{m.identity_shorter:.6f}",
                    f"{m.query_coverage:.6f}", f"{m.subject_coverage:.6f}",
                    m.aln_len, m.score,
                ])
    logging.info(f"Results exported to {output_file}")
