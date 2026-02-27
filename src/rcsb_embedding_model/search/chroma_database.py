import chromadb
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple


class ChromaEmbeddingDatabase:
    """ChromaDB-based database for structure chain embeddings."""

    def __init__(self, db_path: str, collection_name: str = "structure_embeddings"):
        """
        Initialize ChromaDB database.

        Args:
            db_path: Path to store the ChromaDB database
            collection_name: Name of the collection to use
        """
        self.db_path = Path(db_path)
        self.collection_name = collection_name
        self.client = None
        self.collection = None

    def create_database(
            self,
            chain_ids: List[str],
            embeddings: List[torch.Tensor]
    ):
        """
        Create a new ChromaDB database from chain embeddings.

        Args:
            chain_ids: List of chain identifiers (format: "structure_name:chain_id")
            embeddings: List of embedding tensors (one per chain)
        """
        if len(chain_ids) != len(embeddings):
            raise ValueError("Number of chain_ids must match number of embeddings")

        # Create persistent client
        self.client = chromadb.PersistentClient(path=str(self.db_path))

        # Delete collection if it exists
        try:
            self.client.delete_collection(name=self.collection_name)
        except:
            pass

        # Create collection
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        # Convert embeddings to list format for storage
        print("Converting embeddings for storage...")
        embedding_list = []
        for embedding in embeddings:
            if isinstance(embedding, torch.Tensor):
                embedding = embedding.detach().cpu().numpy()
            # Embeddings should already be protein-level (1D vectors)
            if embedding.ndim > 1:
                # If still multi-dimensional, flatten or take mean
                embedding = np.mean(embedding, axis=0)
            embedding_list.append(embedding.tolist())

        # Add to collection in batches
        batch_size = 1000
        total = len(chain_ids)
        print(f"Adding {total} chains to ChromaDB...")

        for i in range(0, total, batch_size):
            end_idx = min(i + batch_size, total)
            batch_ids = chain_ids[i:end_idx]
            batch_embeddings = embedding_list[i:end_idx]

            self.collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                metadatas=[{"chain_id": cid} for cid in batch_ids]
            )
            print(f"  Added {end_idx}/{total} chains")

        print("Database created successfully!")

    def load_database(self):
        """Load an existing ChromaDB database."""
        if not self.db_path.exists():
            raise ValueError(f"Database path does not exist: {self.db_path}")

        self.client = chromadb.PersistentClient(path=str(self.db_path))
        self.collection = self.client.get_collection(name=self.collection_name)
        print(f"Loaded database with {self.collection.count()} chains")

    def search(
            self,
            query_embedding: torch.Tensor,
            top_k: int = 10
    ) -> Tuple[List[str], List[float]]:
        """
        Search the database for similar chains.

        Args:
            query_embedding: Query embedding tensor (protein-level)
            top_k: Number of top results to return

        Returns:
            Tuple of (chain_ids, distances) for the top matches
        """
        if self.collection is None:
            raise ValueError("Database not loaded. Call load_database() first.")

        # Convert query embedding to list
        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.detach().cpu().numpy()
        # Embeddings should already be protein-level (1D vectors)
        if query_embedding.ndim > 1:
            # If still multi-dimensional, flatten or take mean
            query_embedding = np.mean(query_embedding, axis=0)
        query_vector = query_embedding.tolist()

        # Query the collection
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k
        )

        chain_ids = results['ids'][0]
        distances = results['distances'][0]

        return chain_ids, distances

    def search_by_chain_ids(
            self,
            query_chain_ids: List[str],
            top_k: int = 10
    ) -> Dict[str, Tuple[List[str], List[float]]]:
        """
        Search database using existing chain IDs.

        Args:
            query_chain_ids: List of chain IDs to use as queries
            top_k: Number of top results per query

        Returns:
            Dictionary mapping query chain ID to (matching_chain_ids, distances)
        """
        if self.collection is None:
            raise ValueError("Database not loaded. Call load_database() first.")

        results_dict = {}
        for chain_id in query_chain_ids:
            try:
                # Get the embedding for this chain
                result = self.collection.get(ids=[chain_id], include=['embeddings'])
                if not result['embeddings']:
                    print(f"Chain {chain_id} not found in database")
                    continue

                query_embedding = result['embeddings'][0]

                # Search with this embedding
                search_results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k
                )

                chain_ids = search_results['ids'][0]
                distances = search_results['distances'][0]
                results_dict[chain_id] = (chain_ids, distances)

            except Exception as e:
                print(f"Error searching for {chain_id}: {e}")
                continue

        return results_dict

    def get_statistics(self) -> Dict:
        """Get database statistics."""
        if self.collection is None:
            raise ValueError("Database not loaded. Call load_database() first.")

        return {
            "total_chains": self.collection.count(),
            "collection_name": self.collection_name,
            "db_path": str(self.db_path)
        }
