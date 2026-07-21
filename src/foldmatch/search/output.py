"""Shared tabular output contract for search results.

Every result file this package writes — the Stage-1 embedding search and the
Stage-2 sequence alignment alike — has the same shape: delimited rows, one hit
per line, no header. Downstream parsing therefore never has to branch on which
stage produced a file, and a column name means the same thing in both.

Field *rendering* for Stage-2's selectable columns lives in :mod:`alignment`,
which owns the alignment internals those columns are derived from. This module
owns the file-level convention (delimiter, no header) and the column set an
embedding-only search emits.
"""

import logging
from typing import Dict, Iterable, List, Tuple

logger = logging.getLogger(__name__)

# Tab-separated, no header row — the shape of every result file.
DELIMITER = "\t"

# Columns emitted by an embedding-only search (i.e. no Stage-2 alignment: the
# structure/embedding queries, and the sequence queries when Stage 2 is off).
# These reuse the Stage-2 field names so the columns line up across both files.
EMBEDDING_OUTPUT_FIELDS: Tuple[str, ...] = ("query", "target", "embscore")

# Columns emitted by ``fm-search cluster``. Clustering describes the database
# rather than a query/target pair, so it has its own column set — but the same
# tab-separated, no-header file convention.
CLUSTER_OUTPUT_FIELDS: Tuple[str, ...] = ("chain_id", "cluster_id", "cluster_size")


def format_embedding_score(score: float) -> str:
    """Render an embedding similarity score.

    Single source of truth, so an embedding-only file and Stage-2's
    ``embscore`` column format the same number identically.
    """
    return f"{score:.6f}"


def write_rows(rows: Iterable[str], output_file: str) -> int:
    """Write pre-rendered rows (no header) to ``output_file``; return the count."""
    n = 0
    with open(output_file, "w", newline="") as fh:
        for row in rows:
            fh.write(row)
            fh.write("\n")
            n += 1
    return n


def write_embedding_results(
        results: Dict[str, Tuple[List[str], List[float]]],
        output_file: str,
        delimiter: str = DELIMITER,
) -> None:
    """Write embedding-search hits as ``query, target, embscore`` rows.

    ``results`` is the ``{query_id: ([target_ids], [scores])}`` mapping the
    embedding search returns. Hits are emitted in the order the search produced
    them (descending similarity), so rank is simply the row order — there is no
    separate rank column.
    """
    def _rows():
        for query_id, (target_ids, scores) in results.items():
            for target_id, score in zip(target_ids, scores):
                yield delimiter.join((query_id, target_id, format_embedding_score(score)))

    n = write_rows(_rows(), output_file)
    logger.info(f"Wrote {n} result row(s) to {output_file}")
