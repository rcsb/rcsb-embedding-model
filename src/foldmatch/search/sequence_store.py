import logging
import sqlite3
from pathlib import Path
from typing import Dict, Iterable

from foldmatch.utils.fasta import iter_fasta

logger = logging.getLogger(__name__)

# Sidecar file that lives next to a FAISS database (``{index_name}.sequences``)
# and holds the id -> sequence mapping used by the Stage-2 pairwise alignment.
SEQUENCE_STORE_SUFFIX = ".sequences"

# SQLite caps the number of host parameters per statement (SQLITE_MAX_VARIABLE_NUMBER,
# historically 999). Stay safely below it when expanding an ``IN (...)`` clause.
_SQLITE_VAR_CHUNK = 900

# Rows buffered between executemany/commit when building the store. Bounds peak
# RAM and journal size during bulk load so the store scales to corpora of
# hundreds of millions of sequences without materializing the FASTA in memory.
_INSERT_BATCH = 50_000


class SequenceStore:
    """SQLite-backed ``id -> sequence`` store sitting next to a FAISS database.

    The store is built directly from the source FASTA at database-build time.
    The FAISS database ids *are* the FASTA headers (both derive from
    :func:`parse_fasta`), so no plumbing through the embedding pipeline is
    needed: a second independent pass over the FASTA reproduces exactly the
    ids present in the index.

    At query time Stage-2 alignment only needs the handful of candidate
    sequences returned by the embedding prefilter, so reads are random-access
    by id (``WHERE id IN (...)``) and never load the whole corpus into RAM —
    which keeps the large on-disk IVF-PQ path viable.
    """

    def __init__(self, db_folder: Path, index_name: str):
        self.db_folder = Path(db_folder)
        self.index_name = index_name
        self.path = self.db_folder / f"{index_name}{SEQUENCE_STORE_SUFFIX}"

    def exists(self) -> bool:
        return self.path.exists()

    def create(self, fasta_file: Path, min_res_n: int = 0, force: bool = False) -> int:
        """Build the store from a FASTA file, streaming to bound memory.

        The FASTA is read one record at a time and inserted in batches, so peak
        RAM is independent of corpus size (scales to hundreds of millions of
        sequences). Applies the same ``min_res_n`` filter used when building the
        index so store and index stay in lock-step. Duplicate headers keep the
        first occurrence — deduplication is delegated to the table's primary key
        via ``INSERT OR IGNORE`` (no in-RAM ``seen`` set).

        Returns the number of sequences stored.
        """
        if self.path.exists():
            if not force:
                raise FileExistsError(
                    f"Sequence store already exists at {self.path}. "
                    f"Choose a different --output-db, or delete the existing file first."
                )
            self.path.unlink()

        self.db_folder.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(self.path))
        try:
            # Bulk-load tuning: the store is a derived artifact (rebuildable from
            # the FASTA), so durability during build is not required. Per-batch
            # commits keep the in-memory rollback journal small.
            conn.execute("PRAGMA journal_mode = MEMORY")
            conn.execute("PRAGMA synchronous = OFF")
            conn.execute(
                "CREATE TABLE sequences ("
                "id TEXT PRIMARY KEY, sequence TEXT NOT NULL, length INTEGER NOT NULL)"
            )

            insert_sql = "INSERT OR IGNORE INTO sequences (id, sequence, length) VALUES (?, ?, ?)"
            seen_input = 0
            batch: list = []
            for name, sequence in iter_fasta(fasta_file):
                sequence = sequence.strip().upper()
                if min_res_n > 0 and len(sequence) < min_res_n:
                    continue
                batch.append((name, sequence, len(sequence)))
                seen_input += 1
                if len(batch) >= _INSERT_BATCH:
                    conn.executemany(insert_sql, batch)
                    conn.commit()
                    batch.clear()
            if batch:
                conn.executemany(insert_sql, batch)
                conn.commit()

            stored = conn.execute("SELECT COUNT(*) FROM sequences").fetchone()[0]
        finally:
            conn.close()

        duplicates = seen_input - stored
        if duplicates > 0:
            logger.warning(
                f"{duplicates} duplicate FASTA id(s) in sequence store; kept first occurrence of each"
            )
        logger.info(f"Sequence store written to {self.path} ({stored} sequences)")
        return stored

    def fetch(self, ids: Iterable[str]) -> Dict[str, str]:
        """Return ``{id: sequence}`` for the given ids that exist in the store.

        Missing ids are simply absent from the returned dict (no error), so
        callers can skip candidates whose sequence is unavailable.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"Sequence store not found at {self.path}")

        unique_ids = list(dict.fromkeys(ids))
        if not unique_ids:
            return {}

        conn = sqlite3.connect(str(self.path))
        try:
            out: Dict[str, str] = {}
            for start in range(0, len(unique_ids), _SQLITE_VAR_CHUNK):
                chunk = unique_ids[start:start + _SQLITE_VAR_CHUNK]
                placeholders = ",".join("?" * len(chunk))
                cur = conn.execute(
                    f"SELECT id, sequence FROM sequences WHERE id IN ({placeholders})",
                    chunk,
                )
                for cid, sequence in cur.fetchall():
                    out[cid] = sequence
            return out
        finally:
            conn.close()

    def count(self) -> int:
        """Number of sequences in the store."""
        if not self.path.exists():
            raise FileNotFoundError(f"Sequence store not found at {self.path}")
        conn = sqlite3.connect(str(self.path))
        try:
            return conn.execute("SELECT COUNT(*) FROM sequences").fetchone()[0]
        finally:
            conn.close()

    def total_residues(self) -> int:
        """Sum of all sequence lengths — the search space for database E-values."""
        if not self.path.exists():
            raise FileNotFoundError(f"Sequence store not found at {self.path}")
        conn = sqlite3.connect(str(self.path))
        try:
            return conn.execute("SELECT COALESCE(SUM(length), 0) FROM sequences").fetchone()[0]
        finally:
            conn.close()
