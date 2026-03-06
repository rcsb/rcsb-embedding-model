import warnings

import torch
from pathlib import Path
from typing import List, Tuple, Dict

from rcsb_embedding_model.rcsb_structure_embedding import RcsbStructureEmbedding
from rcsb_embedding_model.search.faiss_database import FaissEmbeddingDatabase
from rcsb_embedding_model.types.api_types import StructureFormat


class StructureSearch:
    """Search for similar protein structures using embeddings."""

    def __init__(
            self,
            db_path: str,
            index_name: str = "structure_embeddings",
            min_res: int = 10,
            max_res: int = None,
            device: torch.device = None,
            use_gpu_for_search: bool = False
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
        """
        self.device = device
        self.min_res = min_res
        self.max_res = max_res
        self.db = FaissEmbeddingDatabase(db_path, index_name)
        self.db.load_database(use_gpu=use_gpu_for_search)
        self.embedder = None

    def _get_embedder(self) -> RcsbStructureEmbedding:
        """Load embedding models only when structure-based search is needed."""
        if self.embedder is None:
            self.embedder = RcsbStructureEmbedding(
                min_res=self.min_res,
                max_res=self.max_res
            )
            self.embedder.load_models(device=self.device)
        return self.embedder

    def search_by_structure(
            self,
            query_structure: str,
            structure_format: StructureFormat = StructureFormat.mmcif,
            chain_id: str = None,
            top_k: int = 10
    ) -> Dict[str, Tuple[List[str], List[float]]]:
        """
        Search database using a structure file.

        Args:
            query_structure: Path to query structure file
            structure_format: Format of structure file
            chain_id: Specific chain to search (if None, searches all chains)
            top_k: Number of top results per chain

        Returns:
            Dictionary mapping query chain ID to (matching_chain_ids, similarity_scores)
        """
        query_path = Path(query_structure)
        if not query_path.exists():
            raise ValueError(f"Query structure file does not exist: {query_structure}")

        structure_name = query_path.stem
        print(f"Processing query structure: {structure_name}")
        embedder = self._get_embedder()

        # Suppress biotite warnings during structure loading
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="biotite")
            warnings.filterwarnings("ignore", category=FutureWarning, module="esm")
            # Get residue-level embeddings for chains in the query structure
            # If chain_id is specified, only compute embeddings for that chain
            chain_residue_embeddings = embedder.residue_embedding_by_chain(
                src_structure=query_structure,
                structure_format=structure_format,
                chain_id=chain_id
            )

        if not chain_residue_embeddings:
            if chain_id:
                raise ValueError(f"Chain {chain_id} not found or does not meet minimum residue requirements")
            else:
                raise ValueError("No valid chains found in query structure")

        results = {}
        for chain_id, residue_embedding in chain_residue_embeddings.items():
            print(f"Searching with chain {chain_id} ({residue_embedding.shape[0]} residues)...")
            # Apply aggregator to get protein-level embedding
            protein_embedding = embedder.aggregator_embedding(residue_embedding)
            matching_ids, scores = self.db.search(protein_embedding, top_k=top_k)
            query_chain_id = f"{structure_name}:{chain_id}"
            results[query_chain_id] = (matching_ids, scores)

        return results

    def search_by_database(
            self,
            query_db_path: str,
            query_index_name: str = "structure_embeddings",
            top_k: int = 10
    ) -> Dict[str, Tuple[List[str], List[float]]]:
        """
        Search the subject database using every chain embedding from another database.

        Args:
            query_db_path: Path to the query FAISS database directory
            query_index_name: Name of the query FAISS index
            top_k: Number of top results to return per query chain

        Returns:
            Dictionary mapping query chain ID to (matching_chain_ids, similarity_scores)
        """
        print("\nLoading query database...")
        query_db = FaissEmbeddingDatabase(query_db_path, query_index_name)
        query_db.load_database()

        print(f"Query database contains {len(query_db.chain_ids)} chains")
        print(f"\nQuerying all {len(query_db.chain_ids)} chains from query database...")

        results = {}
        for chain_idx, query_chain_id in enumerate(query_db.chain_ids, 1):
            query_embedding = torch.from_numpy(query_db.index.reconstruct(chain_idx - 1))
            matching_ids, scores = self.db.search(query_embedding, top_k=top_k)
            results[query_chain_id] = (matching_ids, scores)

            if chain_idx % 100 == 0:
                print(f"  Processed {chain_idx}/{len(query_db.chain_ids)} queries...")

        print(f"Completed {len(results)} queries")
        return results

    def print_results(self, results: Dict[str, Tuple[List[str], List[float]]]):
        """
        Pretty print search results.

        Args:
            results: Dictionary from search_by_structure
        """
        for query_chain, (matching_ids, scores) in results.items():
            print(f"\n{'='*80}")
            print(f"Query: {query_chain}")
            print(f"{'='*80}")
            if not matching_ids:
                print("No results found matching the criteria")
            else:
                print(f"{'Rank':<6} {'Chain ID':<40} {'Score':<10}")
                print(f"{'-'*80}")
                for rank, (chain_id, score) in enumerate(zip(matching_ids, scores), 1):
                    print(f"{rank:<6} {chain_id:<40} {score:<10.6f}")

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
            writer.writerow(['Query Chain', 'Rank', 'Matching Chain', 'Score'])

            for query_chain, (matching_ids, scores) in results.items():
                for rank, (chain_id, score) in enumerate(zip(matching_ids, scores), 1):
                    writer.writerow([query_chain, rank, chain_id, score])

        print(f"\nResults exported to {output_file}")

    def get_db_statistics(self) -> Dict:
        """Get database statistics."""
        return self.db.get_statistics()
