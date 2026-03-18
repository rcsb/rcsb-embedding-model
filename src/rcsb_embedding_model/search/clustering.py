import logging
import numpy as np
import faiss
import igraph as ig
import leidenalg
from typing import Dict, List, Optional
from pathlib import Path
from tqdm import tqdm

from rcsb_embedding_model.search.faiss_database import FaissEmbeddingDatabase

logger = logging.getLogger(__name__)

class EmbeddingClusterer:
    """Cluster protein structure embeddings using Leiden algorithm on similarity graphs."""

    def __init__(self, db_path: str, index_name: str = "structure_embeddings"):
        """
        Initialize clustering with a FAISS database.

        Args:
            db_path: Path to FAISS database directory
            index_name: Name of the FAISS index
        """
        self.db = FaissEmbeddingDatabase(db_path, index_name)
        self.graph = None
        self.clusters = None
        self.chain_ids = None

    def load_database(self, use_gpu: bool = False):
        """
        Load the FAISS database.

        Args:
            use_gpu: Whether to use GPU for FAISS operations
        """
        self.db.load_database(use_gpu=use_gpu)
        self.chain_ids = self.db.chain_ids

    def build_similarity_graph(
            self,
            threshold: float = 0.8,
            max_neighbors: int = 1000
    ) -> ig.Graph:
        """
        Build a similarity graph using FAISS k-NN search with threshold filtering.

        Args:
            threshold: Minimum similarity score to create an edge (0-1, where 1.0 = identical)
            max_neighbors: Maximum number of neighbors to consider per node

        Returns:
            igraph Graph object with edges weighted by similarity
        """
        if self.db.index is None:
            raise ValueError("Database not loaded. Call load_database() first.")

        n_chains = len(self.chain_ids)
        logging.info(f"\nBuilding similarity graph with threshold >= {threshold}")
        logging.info(f"Searching up to {max_neighbors} neighbors per chain")

        # Clamp max_neighbors to actual database size
        k = min(max_neighbors, n_chains)

        # Get CPU index for reconstruction if using GPU
        index_to_use = self.db.index
        if self.db.is_gpu_index:
            logging.info("Moving GPU index to CPU for reconstruction...")
            index_to_use = faiss.index_gpu_to_cpu(self.db.index)

        edges = []
        weights = []

        # Search incrementally to avoid segfault with large batch searches
        # Both IndexFlatIP and IndexHNSWFlat support reconstruction through their storage
        logging.info("Performing k-NN search ...")
        for i in tqdm(range(n_chains), desc="Processing chains", unit="chain"):
            # Reconstruct single embedding
            query_embedding = index_to_use.reconstruct(i)

            # Ensure contiguous C-order array with correct shape
            query_embedding = np.ascontiguousarray(query_embedding, dtype=np.float32).reshape(1, -1)

            # Search for k nearest neighbors
            scores_i, indices_i = self.db.index.search(query_embedding, k)

            # Build edges from this query
            for j, score in zip(indices_i[0], scores_i[0]):
                # Skip self-loops and edges below threshold
                if i < j and score >= threshold:  # i < j to avoid duplicate edges
                    edges.append((i, j))
                    weights.append(float(score))

        logging.info(f"Created graph with {n_chains} nodes and {len(edges)} edges")

        # Create igraph from edge list
        self.graph = ig.Graph(n=n_chains, edges=edges, directed=False)
        self.graph.es['weight'] = weights

        # Add chain IDs as vertex attribute
        self.graph.vs['name'] = self.chain_ids

        return self.graph

    def cluster_leiden(
            self,
            resolution: float = 1.0,
            n_iterations: int = -1,
            seed: Optional[int] = None
    ) -> List[int]:
        """
        Apply Leiden clustering algorithm to the similarity graph.

        Args:
            resolution: Resolution parameter (higher = more clusters, default: 1.0)
            n_iterations: Number of iterations (-1 for until convergence)
            seed: Random seed for reproducibility

        Returns:
            List of cluster IDs (one per chain)
        """
        if self.graph is None:
            raise ValueError("Graph not built. Call build_similarity_graph() first.")

        logging.info(f"Running Leiden clustering (resolution={resolution})...")

        # Use RBConfigurationVertexPartition with weights
        # Higher edge weights (higher similarity) encourage nodes to be in same cluster
        partition = leidenalg.find_partition(
            self.graph,
            leidenalg.RBConfigurationVertexPartition,
            weights='weight',
            resolution_parameter=resolution,
            n_iterations=n_iterations,
            seed=seed
        )

        self.clusters = partition.membership
        logging.info(f"Found {len(set(self.clusters))} clusters")

        return self.clusters

    def get_cluster_statistics(self) -> Dict:
        """
        Get statistics about the clustering results.

        Returns:
            Dictionary with clustering statistics
        """
        if self.clusters is None:
            raise ValueError("Clustering not performed. Call cluster_leiden() first.")

        cluster_sizes = {}
        for cluster_id in self.clusters:
            cluster_sizes[cluster_id] = cluster_sizes.get(cluster_id, 0) + 1

        sizes = list(cluster_sizes.values())
        n_clusters = len(cluster_sizes)
        n_singletons = sum(1 for s in sizes if s == 1)

        # Calculate modularity if graph exists
        modularity = None
        if self.graph is not None:
            modularity = self.graph.modularity(self.clusters, weights='weight')

        stats = {
            'n_chains': len(self.clusters),
            'n_clusters': n_clusters,
            'n_singletons': n_singletons,
            'min_cluster_size': min(sizes),
            'max_cluster_size': max(sizes),
            'mean_cluster_size': np.mean(sizes),
            'median_cluster_size': np.median(sizes),
            'modularity': modularity
        }

        return stats

    def export_clusters(
            self,
            output_file: str,
            format: str = 'csv',
            min_cluster_size: Optional[int] = None
    ):
        """
        Export cluster assignments to file.

        Args:
            output_file: Path to output file
            format: Output format ('csv' or 'json')
            min_cluster_size: Only include clusters with at least this many members
        """
        if self.clusters is None:
            raise ValueError("Clustering not performed. Call cluster_leiden() first.")

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Calculate cluster sizes
        cluster_sizes = {}
        for cluster_id in self.clusters:
            cluster_sizes[cluster_id] = cluster_sizes.get(cluster_id, 0) + 1

        # Prepare data
        data = []
        for chain_id, cluster_id in zip(self.chain_ids, self.clusters):
            size = cluster_sizes[cluster_id]
            if min_cluster_size is None or size >= min_cluster_size:
                data.append({
                    'chain_id': chain_id,
                    'cluster_id': cluster_id,
                    'cluster_size': size
                })

        if format == 'csv':
            import csv
            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['chain_id', 'cluster_id', 'cluster_size'])
                writer.writeheader()
                writer.writerows(data)
            logging.info(f"\nCluster assignments exported to {output_file}")

        elif format == 'json':
            import json
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            logging.info(f"\nCluster assignments exported to {output_file}")

        else:
            raise ValueError(f"Unsupported format: {format}. Use 'csv' or 'json'.")

        if min_cluster_size:
            filtered_count = len(data)
            total_count = len(self.clusters)
            logging.info(f"Exported {filtered_count}/{total_count} chains (min_cluster_size={min_cluster_size})")

    def print_statistics(self):
        """Print clustering statistics to console."""
        stats = self.get_cluster_statistics()
        logging.info("CLUSTERING STATISTICS")
        logging.info(f"Total embeddings:    {stats['n_chains']}")
        logging.info(f"Number of clusters:  {stats['n_clusters']}")
        logging.info(f"Singleton clusters:  {stats['n_singletons']}")
        logging.info(f"Cluster size (min):  {stats['min_cluster_size']}")
        logging.info(f"Cluster size (max):  {stats['max_cluster_size']}")
        logging.info(f"Cluster size (mean): {stats['mean_cluster_size']:.2f}")
        logging.info(f"Cluster size (med):  {stats['median_cluster_size']:.1f}")
        if stats['modularity'] is not None:
            logging.info(f"Modularity:          {stats['modularity']:.4f}")
