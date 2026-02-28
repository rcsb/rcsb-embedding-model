import faiss
import torch
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional


def _has_gpu_support():
    """Check if FAISS has GPU support available."""
    try:
        return hasattr(faiss, 'StandardGpuResources') and faiss.get_num_gpus() > 0
    except:
        return False


class FaissEmbeddingDatabase:
    """FAISS-based database for structure chain embeddings."""

    def __init__(self, db_path: str, index_name: str = "structure_embeddings"):
        """
        Initialize FAISS database.

        Args:
            db_path: Path to store the FAISS database
            index_name: Name of the index (used for file naming)
        """
        self.db_path = Path(db_path)
        self.index_name = index_name
        self.index = None
        self.chain_ids = None
        self.dimension = None
        self.gpu_resources = None
        self.is_gpu_index = False

    def create_database(
            self,
            chain_ids: List[str],
            embeddings: List[torch.Tensor],
            use_gpu: bool = False
    ):
        """
        Create a new FAISS database from chain embeddings.

        Args:
            chain_ids: List of chain identifiers (format: "structure_name:chain_id")
            embeddings: List of embedding tensors (one per chain)
            use_gpu: Whether to use GPU for indexing (if available)
        """
        if len(chain_ids) != len(embeddings):
            raise ValueError("Number of chain_ids must match number of embeddings")

        # Convert embeddings to numpy array
        print("Converting embeddings to numpy array...")
        embedding_array = []
        for embedding in embeddings:
            if isinstance(embedding, torch.Tensor):
                embedding = embedding.detach().cpu().numpy()
            # Ensure 1D
            if embedding.ndim > 1:
                embedding = np.mean(embedding, axis=0)
            embedding_array.append(embedding)

        embedding_array = np.array(embedding_array, dtype=np.float32)
        self.dimension = embedding_array.shape[1]
        n_embeddings = embedding_array.shape[0]

        print(f"Creating FAISS index with {n_embeddings} embeddings of dimension {self.dimension}...")

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embedding_array)

        # Choose index type based on dataset size
        if n_embeddings < 10000:
            # Small dataset: use exact search (IndexFlatIP)
            print("Using IndexFlatIP (exact search) for small dataset")
            self.index = faiss.IndexFlatIP(self.dimension)
        else:
            # Large dataset: use HNSW for approximate search
            print("Using IndexHNSWFlat (approximate search) for large dataset")
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)

        # Move to GPU if requested and available
        if use_gpu:
            if _has_gpu_support():
                print(f"Moving index to GPU (detected {faiss.get_num_gpus()} GPU(s))...")
                self.gpu_resources = faiss.StandardGpuResources()
                # Allow up to 1GB of temporary memory for GPU operations
                self.gpu_resources.setTempMemory(1024 * 1024 * 1024)
                self.index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, self.index)
                self.is_gpu_index = True
                print("Index successfully moved to GPU")
            else:
                print("WARNING: GPU requested but not available. Using CPU instead.")
                print("To enable GPU support, install: pip install faiss-gpu")

        # Add vectors to index
        self.index.add(embedding_array)
        self.chain_ids = chain_ids

        # Save to disk
        self._save()

        print(f"Database created successfully with {len(chain_ids)} chains!")

    def load_database(self, use_gpu: bool = False):
        """
        Load an existing FAISS database.

        Args:
            use_gpu: Whether to load the index to GPU (if available)
        """
        index_file = self.db_path / f"{self.index_name}.index"
        metadata_file = self.db_path / f"{self.index_name}.metadata"

        if not index_file.exists() or not metadata_file.exists():
            raise ValueError(f"Database files not found in: {self.db_path}")

        # Load FAISS index (always loads to CPU first)
        self.index = faiss.read_index(str(index_file))

        # Load metadata
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
            self.chain_ids = metadata['chain_ids']
            self.dimension = metadata['dimension']

        print(f"Loaded database with {len(self.chain_ids)} chains")

        # Move to GPU if requested
        if use_gpu:
            self.move_to_gpu()

    def move_to_gpu(self, gpu_id: int = 0):
        """
        Move the index to GPU.

        Args:
            gpu_id: GPU device ID to use (default: 0)
        """
        if self.is_gpu_index:
            print("Index is already on GPU")
            return

        if not _has_gpu_support():
            print("WARNING: GPU not available. Keeping index on CPU.")
            print("To enable GPU support, install: pip install faiss-gpu")
            return

        if self.index is None:
            raise ValueError("No index loaded. Call load_database() first.")

        print(f"Moving index to GPU {gpu_id}...")
        self.gpu_resources = faiss.StandardGpuResources()
        self.gpu_resources.setTempMemory(1024 * 1024 * 1024)  # 1GB temp memory
        self.index = faiss.index_cpu_to_gpu(self.gpu_resources, gpu_id, self.index)
        self.is_gpu_index = True
        print("Index successfully moved to GPU")

    def move_to_cpu(self):
        """Move the index from GPU to CPU."""
        if not self.is_gpu_index:
            print("Index is already on CPU")
            return

        print("Moving index to CPU...")
        self.index = faiss.index_gpu_to_cpu(self.index)
        self.is_gpu_index = False
        self.gpu_resources = None
        print("Index successfully moved to CPU")

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
        if self.index is None:
            raise ValueError("Database not loaded. Call load_database() first.")

        # Convert query embedding to numpy
        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.detach().cpu().numpy()

        # Ensure 1D and correct shape
        if query_embedding.ndim > 1:
            query_embedding = np.mean(query_embedding, axis=0)

        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)

        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)

        # Search
        # Note: FAISS returns inner product (higher is better)
        # For cosine similarity after normalization, distance = 1 - inner_product
        distances, indices = self.index.search(query_embedding, top_k)

        # Convert distances to more intuitive format (cosine distance: 0 = identical, 2 = opposite)
        # Inner product ranges from -1 to 1 after normalization
        # Cosine distance = 1 - inner_product, so ranges from 0 (identical) to 2 (opposite)
        cosine_distances = 1.0 - distances[0]

        # Get chain IDs for the results
        result_chain_ids = [self.chain_ids[idx] for idx in indices[0]]

        return result_chain_ids, cosine_distances.tolist()

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
        if self.index is None:
            raise ValueError("Database not loaded. Call load_database() first.")

        results_dict = {}
        for query_chain_id in query_chain_ids:
            try:
                # Find the index of this chain
                if query_chain_id not in self.chain_ids:
                    print(f"Chain {query_chain_id} not found in database")
                    continue

                chain_idx = self.chain_ids.index(query_chain_id)

                # Get the embedding from the index
                query_embedding = self.index.reconstruct(chain_idx).reshape(1, -1)

                # Search
                distances, indices = self.index.search(query_embedding, top_k)
                cosine_distances = 1.0 - distances[0]

                result_chain_ids = [self.chain_ids[idx] for idx in indices[0]]
                results_dict[query_chain_id] = (result_chain_ids, cosine_distances.tolist())

            except Exception as e:
                print(f"Error searching for {query_chain_id}: {e}")
                continue

        return results_dict

    def get_statistics(self) -> Dict:
        """Get database statistics."""
        if self.index is None:
            raise ValueError("Database not loaded. Call load_database() first.")

        return {
            "total_chains": len(self.chain_ids),
            "dimension": self.dimension,
            "index_name": self.index_name,
            "db_path": str(self.db_path),
            "index_type": type(self.index).__name__,
            "on_gpu": self.is_gpu_index,
            "gpu_available": _has_gpu_support()
        }

    def _save(self):
        """Save FAISS index and metadata to disk."""
        # Create directory if it doesn't exist
        self.db_path.mkdir(parents=True, exist_ok=True)

        index_file = self.db_path / f"{self.index_name}.index"
        metadata_file = self.db_path / f"{self.index_name}.metadata"

        # Save FAISS index
        # If GPU index, convert to CPU before saving
        if hasattr(self.index, 'index'):  # GPU index wrapper
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, str(index_file))
        else:
            faiss.write_index(self.index, str(index_file))

        # Save metadata
        metadata = {
            'chain_ids': self.chain_ids,
            'dimension': self.dimension
        }
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)

        print(f"Saved index to {index_file}")
        print(f"Saved metadata to {metadata_file}")
