import logging
from pathlib import Path
from typing import Dict, List, Tuple, Iterable
from tqdm import tqdm
import numpy as np
import time

from foldmatch.search.faiss_database import FaissEmbeddingDatabase

logger = logging.getLogger(__name__)

class EmbeddingDatabase:
    """Search for similar protein structures using embeddings."""

    def __init__(
            self,
            db_path: str,
            use_gpu_for_search: bool = False
    ):
        """
        Initialize structure search.
        Args:
            db_path: Path to FAISS database
            use_gpu_for_search: Whether to use GPU for FAISS search operations
        """
        self.db_folder, self.index_name, self.db_path = _parse_db_path(db_path)
        self.db = FaissEmbeddingDatabase(self.db_folder, self.index_name)
        self._load_db(use_gpu=use_gpu_for_search)

    def _load_db(self, use_gpu=False):
        index_file =  Path(f"{self.db_path}.index)")
        metadata_file = Path(f"{self.db_path}.metadata")
        if index_file.exists() or metadata_file.exists():
            self.db.load_database(use_gpu=use_gpu)

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

