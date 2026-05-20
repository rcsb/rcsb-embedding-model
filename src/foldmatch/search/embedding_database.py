import logging
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterator, Iterable
from tqdm import tqdm
import numpy as np
import pyarrow.parquet as pq
import pandas as pd
import time

from foldmatch.foldmatch import FoldMatch
from foldmatch.search.embedding_computer import (
    EmbeddingComputer,
)
from foldmatch.search.faiss_database import FaissEmbeddingDatabase
from foldmatch.types.api_types import Accelerator, Granularity, StructureFormat

logger = logging.getLogger(__name__)

class EmbeddingDatabase:
    """Search for similar protein structures using embeddings."""

    def __init__(
            self,
            db_path: str,
            min_res: int = 10,
            max_res: int = None,
            device: torch.device = None,
            use_gpu_for_search: bool = False,
            tmp_dir: Optional[str] = None,
            accelerator: Accelerator = 'auto',
    ):
        """
        Initialize structure search.

        Args:
            db_path: Path to FAISS database
            index_name: Name of the FAISS index
            min_res: Minimum residue length for chains
            max_res: Maximum residue length for structures
            device: Device to use for embedding computation
            use_gpu_for_search: Whether to use GPU for FAISS search operations
            tmp_dir: Scratch directory for FASTA-based search (required by
                :meth:`search_by_fasta`; optional otherwise).
            accelerator: Lightning accelerator for FASTA-based search inference.
        """
        self.device = device
        self.min_res = min_res
        self.max_res = max_res
        self.tmp_dir = tmp_dir
        self.accelerator = accelerator
        self.db_folder, self.index_name, self.db_path = _parse_db_path(db_path)
        self.db = FaissEmbeddingDatabase(self.db_folder, self.index_name)
        self._load_db(use_gpu=use_gpu_for_search)
        self.embedder = None
        self.computer: Optional[EmbeddingComputer] = None

    def _load_db(self, use_gpu=False):
        index_file =  Path(f"{self.db_path}.index)")
        metadata_file = Path(f"{self.db_path}.metadata")
        if index_file.exists() or metadata_file.exists():
            self.db.load_database(use_gpu=use_gpu)

    def _get_embedder(self) -> FoldMatch:
        """Load embedding models only when structure-based search is needed."""
        if self.embedder is None:
            self.embedder = FoldMatch(
                min_res=self.min_res,
                max_res=self.max_res
            )
            self.embedder.load_models(device=self.device)
        return self.embedder

    def _get_embedding_computer(self) -> EmbeddingComputer:
        """Create the FASTA embedding computer lazily; requires ``tmp_dir``."""
        if self.computer is None:
            if self.tmp_dir is None:
                raise ValueError(
                    "tmp_dir must be set on EmbeddingSearch to compute embeddings from FASTA"
                )
            self.computer = EmbeddingComputer(
                embedding_folder=self.tmp_dir, accelerator=self.accelerator
            )
        return self.computer

    def create_db(
            self,
            embedding_batches,
            use_gpu_index,
            index_type,
            index_config
    ):
        start_time = time.time()
        self.db.create_database(
            embedding_batches=embedding_batches,
            index_type=index_type,
            index_config=index_config,
            use_gpu=use_gpu_index,
        )
        database_time = time.time() - start_time
        logging.info(f"Creating database completed in {database_time:.2f} seconds")

        logging.info("Database build complete!")
        logging.info(f"Database location: {self.db_path}")
        logging.info(f"Total embeddings: {len(self.db.chain_ids)}")
        logging.info(f"You can now search this database using:")
        logging.info(f"   fm-search query structure --db-path {self.db_path} --query-structure <path_to_structure>")

    def search_by_embeddings(
            self,
            embedding_batches: Iterable[Tuple[List[str], np.ndarray]],
            top_k: int = 10,
    ):
        results: Dict[str, Tuple[List[str], List[float]]] = {}
        for ids_batch, emb_batch in embedding_batches:
            batch_results = self.db.search_batch(emb_batch, top_k=top_k)
            for qid, res in zip(ids_batch, batch_results):
                results[qid] = res
        return results

    #TODO this method need to be moved to EmbeddingComputer and return a batch of embeddings.
    def search_by_structure(
            self,
            query_structure: str,
            structure_format: StructureFormat = StructureFormat.mmcif,
            granularity: Granularity = 'chain',
            chain_id: str = None,
            assembly_id: str = None,
            top_k: int = 10
    ) -> Dict[str, Tuple[List[str], List[float]]]:
        """
        Search database using a structure file.

        Args:
            query_structure: Path to query structure file
            structure_format: Format of structure file
            granularity: Level of granularity for search ('chain' or 'assembly')
            chain_id: Specific chain to search (if None, searches all chains)
            assembly_id: Specific assembly to search (if None, searches by asymmetric unit)
            top_k: Number of top results per chain

        Returns:
            Dictionary mapping query chain ID to (matching_chain_ids, similarity_scores)
        """
        query_path = Path(query_structure)
        if not query_path.exists():
            raise ValueError(f"Query structure file does not exist: {query_structure}")

        if granularity != 'chain' and granularity != 'assembly':
            raise ValueError(f"Unknown granularity value {granularity}")

        structure_name = query_path.stem
        logging.info(f"Processing residue embeddings: {structure_name}")
        embedder = self._get_embedder()

        residue_embeddings = embedder.residue_embedding_by_chain(
            src_structure=query_structure,
            structure_format=structure_format,
            chain_id=chain_id
        ) if granularity == 'chain' else embedder.residue_embedding_by_assembly(
            src_structure=query_structure,
            structure_format=structure_format,
            assembly_id=assembly_id
        )

        if not residue_embeddings:
            if chain_id:
                raise ValueError(f"Chain {chain_id} not found or does not meet minimum residue requirements")
            elif assembly_id:
                raise ValueError(f"Assembly {assembly_id} not found or does not meet minimum residue requirements")
            else:
                raise ValueError("No valid chains found in query structure")

        results = {}
        for key, residue_embedding in residue_embeddings.items():
            logging.info(f"Searching {structure_name} {granularity.value if hasattr(granularity, 'value') else granularity} {key} ({residue_embedding.shape[0]} residues)...")
            # Apply aggregator to get protein-level embedding
            protein_embedding = embedder.aggregator_embedding(residue_embedding)
            matching_ids, scores = self.db.search(protein_embedding, top_k=top_k)
            query_embedding_id = f"{structure_name}{"." if granularity == 'chain' else "-"}{key}"
            results[query_embedding_id] = (matching_ids, scores)

        return results

    def search_by_database(
            self,
            query_db_path: str,
            top_k: int = 10,
            batch_size: int = 4096,
    ) -> Dict[str, Tuple[List[str], List[float]]]:
        """
        Search the subject database using every chain embedding from another database.

        Args:
            query_db_path: Path to the query FAISS database directory
            query_index_name: Name of the query FAISS index
            top_k: Number of top results to return per query chain
            batch_size: Number of query vectors processed per FAISS call.

        Returns:
            Dictionary mapping query chain ID to (matching_chain_ids, similarity_scores)
        """
        logging.info("Loading query database...")
        query_db_folder, query_index_name, _ = _parse_db_path(query_db_path)
        query_db = FaissEmbeddingDatabase(query_db_folder, query_index_name)
        query_db.load_database()

        n = len(query_db.chain_ids)
        logging.info(f"Query database contains {n} embeddings")

        results = {}
        with tqdm(total=n, desc="Querying database") as pbar:
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch_ids = query_db.chain_ids[start:end]
                batch_vecs = query_db.index.reconstruct_n(start, end - start)
                batch_results = self.db.search_batch(batch_vecs, top_k=top_k)
                for qid, res in zip(batch_ids, batch_results):
                    results[qid] = res
                pbar.update(end - start)

        logging.info(f"Completed {len(results)} queries")
        return results

    def print_results(self, results: Dict[str, Tuple[List[str], List[float]]]):
        """
        Pretty print search results.

        Args:
            results: Dictionary from search_by_structure
        """
        for query_chain, (matching_ids, scores) in results.items():
            logging.info(f"Query: {query_chain}")
            if not matching_ids:
                logging.info("No results found matching the criteria")
            else:
                logging.info(f"{'Rank':<6} {'Match':<40} {'Score':<10}")
                for rank, (chain_id, score) in enumerate(zip(matching_ids, scores), 1):
                    logging.info(f"{rank:<6} {chain_id:<40} {score:<10.6f}")

    def export_results(
            self,
            results: Dict[str, Tuple[List[str], List[float]]],
            output_file: str
    ):
        """
        Export search results to a CSV file.

        Args:
            results: Dictionary from search_by_structure
            output_file: Path to output CSV file
        """
        import csv

        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Query', 'Rank', 'Match', 'Score'])

            for query_chain, (matching_ids, scores) in results.items():
                for rank, (chain_id, score) in enumerate(zip(matching_ids, scores), 1):
                    writer.writerow([query_chain, rank, chain_id, score])

        logging.info(f"Results exported to {output_file}")

    def get_db_statistics(self) -> Dict:
        """Get database statistics."""
        return self.db.get_statistics()


def _parse_db_path(output_db: str) -> tuple[Path, str, str]:
    """Split a database path into (directory, index name, resolved path)."""
    output_db_path = Path(output_db)
    db_dir = output_db_path.parent
    index_name = output_db_path.name or "embeddings"
    if db_dir == Path('.'):
        db_dir = Path.cwd()
    return db_dir, index_name, str(db_dir / index_name)


_POINT_EXTS = ('.pt', '.csv')
_BATCH_EXTS = ('.parquet',)
_SUPPORTED_EXTS = _POINT_EXTS + _BATCH_EXTS

def stream_embeddings(
        path: str,
        file_extension: Optional[str] = None,
        batch_size: int = 4096,
) -> Iterator[tuple[list[str], np.ndarray]]:
    """Yield (ids, [B, D] float32) batches from a file or directory of .pt / .csv / .parquet.

    Parquet shards are streamed via ``ParquetFile.iter_batches`` (many chains per file).
    .pt and .csv hold one chain per file (ID = filename stem) and are chunked into
    batches of ``batch_size``. A single file is handled as a degenerate directory-of-one;
    ``file_extension`` is ignored in that case.
    """
    p = Path(path)
    if not p.exists():
        raise ValueError(f"Embeddings path does not exist: {path}")

    if file_extension is not None and file_extension not in _SUPPORTED_EXTS:
        raise ValueError(
            f"Unsupported file extension '{file_extension}'. "
            f"Use one of: {', '.join(_SUPPORTED_EXTS)}"
        )

    if p.is_file():
        if p.suffix not in _SUPPORTED_EXTS:
            raise ValueError(
                f"Unsupported file extension '{p.suffix}'. "
                f"Use one of: {', '.join(_SUPPORTED_EXTS)}"
            )
        files = [p]
    else:
        extensions = (file_extension,) if file_extension is not None else _SUPPORTED_EXTS
        files = []
        for ext in extensions:
            files.extend(sorted(p.glob(f"*{ext}")))
        if not files:
            raise ValueError(
                f"No embedding files found with extensions {list(extensions)} in {path}"
            )

    parquet_files = [f for f in files if f.suffix in _BATCH_EXTS]
    point_files = [f for f in files if f.suffix in _POINT_EXTS]

    for parquet_file in parquet_files:
        pf = pq.ParquetFile(parquet_file)
        for record in pf.iter_batches(batch_size=batch_size, columns=['id', 'embedding']):
            ids = record.column('id').to_pylist()
            flat = record.column('embedding').values.to_numpy(zero_copy_only=False)
            arr = np.ascontiguousarray(
                flat.reshape(len(record), -1).astype(np.float32, copy=False)
            )
            yield ids, arr

    ids_buf: list[str] = []
    emb_buf: list[np.ndarray] = []
    for f in point_files:
        if f.suffix == '.pt':
            emb = torch.load(f, map_location='cpu', weights_only=True)
            if isinstance(emb, torch.Tensor):
                emb = emb.detach().cpu().numpy()
        else:  # .csv
            emb = pd.read_csv(f, header=None).values
        emb = np.asarray(emb, dtype=np.float32)
        if emb.ndim > 1:
            emb = np.mean(emb, axis=0)
        ids_buf.append(f.stem)
        emb_buf.append(emb.reshape(-1))
        if len(ids_buf) >= batch_size:
            yield ids_buf, np.ascontiguousarray(np.stack(emb_buf))
            ids_buf, emb_buf = [], []
    if ids_buf:
        yield ids_buf, np.ascontiguousarray(np.stack(emb_buf))
