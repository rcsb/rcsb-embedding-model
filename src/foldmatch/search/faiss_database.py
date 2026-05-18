import logging
import faiss
import torch
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from foldmatch.types.api_types import IndexConfig, IndexType

logger = logging.getLogger(__name__)

def _has_gpu_support():
    """Check if FAISS has GPU support available."""
    try:
        return hasattr(faiss, 'StandardGpuResources') and faiss.get_num_gpus() > 0
    except:
        return False


def _normalize_L2_batched(embedding_array: np.ndarray, batch_size: int = 1000):
    """Normalize embeddings in-place by batches to avoid segfaults on very large arrays."""
    n = embedding_array.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        faiss.normalize_L2(embedding_array[start:end])


def _index_add_batched(index, embedding_array: np.ndarray, batch_size: int = 100):
    """Add embeddings to a FAISS index by batches to avoid segfaults on very large arrays."""
    n = embedding_array.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        index.add(np.ascontiguousarray(embedding_array[start:end]))


class FaissEmbeddingDatabase:
    """FAISS-based database for structure chain embeddings."""

    N_EMBEDDINGS_DB_THR = 10000

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
        self.index_type = None
        self.index_config: Optional[IndexConfig] = None

    def create_database(
            self,
            batches: Iterable[Tuple[List[str], np.ndarray]],
            index_type: IndexType = IndexType.auto,
            index_config: Optional[IndexConfig] = None,
            use_gpu: bool = False,
    ):
        """
        Create a new FAISS database from a stream of embedding batches.

        Args:
            batches: Iterable yielding ``(ids, [B, D] float32 ndarray)`` tuples.
            index_type: FAISS index variant. See :class:`IndexType`.
            index_config: Tuning knobs (see :class:`IndexConfig`). Defaults if None.
            use_gpu: Move the in-memory index to GPU after build (ignored for ivf_pq).
        """
        config = index_config or IndexConfig()
        self.index_config = config
        self.index_type = index_type
        if index_type == IndexType.ivf_pq:
            if use_gpu:
                logger.warning("use_gpu is not supported for ivf_pq builds; running on CPU.")
            self._create_ivf_pq_ondisk(batches, config)
        else:
            self._create_in_memory(batches, index_type, config, use_gpu)
        self._save()

    def _create_in_memory(
            self,
            batches: Iterable[Tuple[List[str], np.ndarray]],
            index_type: IndexType,
            config: IndexConfig,
            use_gpu: bool,
    ):
        """Build an in-memory FlatIP/HNSW index by materializing all batches."""
        chain_ids: list[str] = []
        arrs: list[np.ndarray] = []
        for ids_batch, emb_batch in batches:
            chain_ids.extend(ids_batch)
            arrs.append(emb_batch)
        if not arrs:
            raise ValueError("No embeddings provided to create_database")
        embedding_array = np.ascontiguousarray(np.concatenate(arrs, axis=0), dtype=np.float32)
        del arrs

        self.dimension = embedding_array.shape[1]
        n_embeddings = embedding_array.shape[0]

        _normalize_L2_batched(embedding_array)

        if index_type == IndexType.auto:
            if n_embeddings < FaissEmbeddingDatabase.N_EMBEDDINGS_DB_THR:
                index_type = IndexType.flat
            else:
                index_type = IndexType.hnsw

        logger.info(f"Building {index_type.name} index with {n_embeddings} embeddings")
        if index_type == IndexType.flat:
            self.index = faiss.IndexFlatIP(self.dimension)
        elif index_type == IndexType.hnsw:
            self.index = faiss.IndexHNSWFlat(
                self.dimension, config.hnsw_m, faiss.METRIC_INNER_PRODUCT
            )

        if use_gpu:
            if _has_gpu_support():
                self.gpu_resources = faiss.StandardGpuResources()
                self.gpu_resources.setTempMemory(1024 * 1024 * 1024)
                self.index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, self.index)
                self.is_gpu_index = True

        _index_add_batched(self.index, embedding_array)
        self.chain_ids = chain_ids

    def _create_ivf_pq_ondisk(
            self,
            batches: Iterable[Tuple[List[str], np.ndarray]],
            config: IndexConfig,
    ):
        """Build OPQ + IVF-PQ with OnDiskInvertedLists. Bounded RAM ≈ training sample size."""

        logger.info(f"Building {IndexType.ivf_pq.name} index")
        self.db_path.mkdir(parents=True, exist_ok=True)
        invlists_path = str(self.db_path / f"{self.index_name}.invlists")

        # Phase A: accumulate the training sample. Keep raw (un-normalized) batches so
        # we can add them to the index after training without double-normalization.
        train_buffer: list[tuple[list[str], np.ndarray]] = []
        n_buffered = 0
        batches_iter = iter(batches)
        for ids_batch, emb_batch in batches_iter:
            train_buffer.append((ids_batch, emb_batch))
            n_buffered += emb_batch.shape[0]
            if n_buffered >= config.opq_train_size:
                break
        if not train_buffer:
            raise ValueError("No embeddings provided to create_database")

        self.dimension = train_buffer[0][1].shape[1]
        if self.dimension % config.m != 0:
            raise ValueError(
                f"PQ subquantizer count m={config.m} must divide embedding dim D={self.dimension}"
            )

        # Phase B: build the empty index, train on a normalized copy of the sample.
        factory_string = f"OPQ{config.m},IVF{config.nlist},PQ{config.m}x{config.nbits}"
        logger.info(
            f"Training {factory_string} on {n_buffered} vectors (cap {config.opq_train_size})"
        )
        self.index = faiss.index_factory(
            self.dimension, factory_string, faiss.METRIC_INNER_PRODUCT
        )

        train_arr = np.ascontiguousarray(
            np.concatenate([a for _, a in train_buffer], axis=0)[:config.opq_train_size],
            dtype=np.float32,
        )
        faiss.normalize_L2(train_arr)
        self.index.train(train_arr)
        del train_arr

        # Phase C: swap inverted lists onto disk so subsequent add() writes go to mmap.
        ivf_index = faiss.extract_index_ivf(self.index)
        invlists = faiss.OnDiskInvertedLists(
            ivf_index.nlist, ivf_index.code_size, invlists_path
        )
        ivf_index.replace_invlists(invlists, True)

        # Phase D: add the training-sample batches, then the rest of the stream.
        self.chain_ids = []
        total_added = 0
        for ids_batch, emb_batch in train_buffer:
            total_added += self._normalize_and_add(ids_batch, emb_batch)
        del train_buffer
        for ids_batch, emb_batch in batches_iter:
            total_added += self._normalize_and_add(ids_batch, emb_batch)

        if config.nprobe is None:
            config.nprobe = max(1, config.nlist // 64)
        ivf_index.nprobe = config.nprobe

        logger.info(f"IVF-PQ build complete: {total_added} vectors, nprobe={config.nprobe}")

    def _normalize_and_add(self, ids_batch: list[str], emb_batch: np.ndarray) -> int:
        """Normalize a copy of the batch and add to the index. Returns the count added."""
        arr = np.ascontiguousarray(emb_batch, dtype=np.float32).copy()
        faiss.normalize_L2(arr)
        self.index.add(arr)
        self.chain_ids.extend(ids_batch)
        return arr.shape[0]

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

        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
        self.chain_ids = metadata['embedding_ids']
        self.dimension = metadata['dimension']
        self.index_type = metadata.get('index_type', IndexType.auto)
        self.index_config = metadata.get('index_config')

        if self.index_type == IndexType.ivf_pq:
            # OnDiskInvertedLists has the absolute path baked in at write time; skip
            # loading them and re-point at the local file so the DB stays portable.
            self.index = faiss.read_index(str(index_file), faiss.IO_FLAG_SKIP_IVF_DATA)
            invlists_path = str(self.db_path / f"{self.index_name}.invlists")
            ivf_index = faiss.extract_index_ivf(self.index)
            new_invlists = faiss.OnDiskInvertedLists(
                ivf_index.nlist, ivf_index.code_size, invlists_path
            )
            ivf_index.replace_invlists(new_invlists, True)
            if self.index_config is not None and self.index_config.nprobe is not None:
                ivf_index.nprobe = self.index_config.nprobe
        else:
            self.index = faiss.read_index(str(index_file))

        logger.info(
            f"Loaded database with {len(self.chain_ids)} embeddings (index_type={self.index_type})"
        )

        if use_gpu:
            self.move_to_gpu()

    def move_to_gpu(self, gpu_id: int = 0):
        """
        Move the index to GPU.

        Args:
            gpu_id: GPU device ID to use (default: 0)
        """
        if self.is_gpu_index:
            logger.info("Index is already on GPU")
            return

        if not _has_gpu_support():
            logger.info("WARNING: GPU not available. Keeping index on CPU.")
            logger.info("To enable GPU support, install: pip install faiss-gpu")
            return

        if self.index is None:
            raise ValueError("No index loaded. Call load_database() first.")

        logger.info(f"Moving index to GPU {gpu_id}...")
        self.gpu_resources = faiss.StandardGpuResources()
        self.gpu_resources.setTempMemory(1024 * 1024 * 1024)  # 1GB temp memory
        self.index = faiss.index_cpu_to_gpu(self.gpu_resources, gpu_id, self.index)
        self.is_gpu_index = True
        logger.info("Index successfully moved to GPU")

    def move_to_cpu(self):
        """Move the index from GPU to CPU."""
        if not self.is_gpu_index:
            logger.info("Index is already on CPU")
            return

        logger.info("Moving index to CPU...")
        self.index = faiss.index_gpu_to_cpu(self.index)
        self.is_gpu_index = False
        self.gpu_resources = None
        logger.info("Index successfully moved to CPU")

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
            Tuple of (chain_ids, similarity_scores) for the top matches.
            Similarity scores range from -1 to 1, where 1.0 indicates a perfect match (similar to TM-score).
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
        # After L2 normalization, inner product equals cosine similarity
        # Cosine similarity ranges from -1 to 1, where 1.0 = identical (like TM-score)
        scores, indices = self.index.search(query_embedding, top_k)

        # Get chain IDs for the results
        result_chain_ids = [self.chain_ids[idx] for idx in indices[0] if idx >= 0]
        result_scores = [scores[0][idx] for idx, jdx in enumerate(indices[0]) if jdx >= 0]

        return result_chain_ids, result_scores

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
            Dictionary mapping query chain ID to (matching_chain_ids, similarity_scores).
            Similarity scores range from -1 to 1, where 1.0 indicates a perfect match (similar to TM-score).
        """
        if self.index is None:
            raise ValueError("Database not loaded. Call load_database() first.")

        results_dict = {}
        for query_chain_id in query_chain_ids:
            try:
                # Find the index of this chain
                if query_chain_id not in self.chain_ids:
                    logger.info(f"Chain {query_chain_id} not found in database")
                    continue

                chain_idx = self.chain_ids.index(query_chain_id)

                # Get the embedding from the index
                query_embedding = self.index.reconstruct(chain_idx).reshape(1, -1)

                # Search
                scores, indices = self.index.search(query_embedding, top_k)

                result_chain_ids = [self.chain_ids[idx] for idx in indices[0]]
                results_dict[query_chain_id] = (result_chain_ids, scores[0].tolist())

            except Exception as e:
                logger.info(f"Error searching for {query_chain_id}: {e}")
                continue

        return results_dict

    def get_statistics(self) -> Dict:
        """Get database statistics."""
        if self.index is None:
            raise ValueError("Database not loaded. Call load_database() first.")

        return {
            "total_embeddings": len(self.chain_ids),
            "dimension": self.dimension,
            "index_name": self.index_name,
            "db_path": str(self.db_path),
            "index_type": type(self.index).__name__,
            "on_gpu": self.is_gpu_index,
            "gpu_available": _has_gpu_support()
        }

    def _save(self):
        """Save FAISS index and metadata to disk."""
        self.db_path.mkdir(parents=True, exist_ok=True)

        index_file = self.db_path / f"{self.index_name}.index"
        metadata_file = self.db_path / f"{self.index_name}.metadata"

        if self.is_gpu_index:
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, str(index_file))
        else:
            faiss.write_index(self.index, str(index_file))

        metadata = {
            'embedding_ids': self.chain_ids,
            'dimension': self.dimension,
            'index_type': self.index_type,
            'index_config': self.index_config,
        }
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)

