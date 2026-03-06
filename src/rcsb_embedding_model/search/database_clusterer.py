import csv
import importlib
from collections import defaultdict
from typing import Dict, List, Tuple

from rcsb_embedding_model.search.faiss_database import FaissEmbeddingDatabase


class DatabaseClusterer:
    """Cluster a FAISS embedding database using Leiden on an approximate kNN graph."""

    def __init__(
            self,
            db_path: str,
            index_name: str = "structure_embeddings",
            use_gpu_for_search: bool = False
    ):
        self.db = FaissEmbeddingDatabase(db_path, index_name)
        self.db.load_database()
        self.use_gpu_for_search = use_gpu_for_search

    def cluster_by_similarity(
            self,
            threshold: float,
            max_neighbors: int = 50,
            resolution: float = 1.0,
            seed: int = 0,
            block_size: int = 1024
    ) -> Dict:
        """
        Cluster database embeddings using Leiden on an approximate kNN similarity graph.

        Args:
            threshold: Minimum cosine similarity required to create a graph edge
            max_neighbors: Number of nearest neighbors to inspect per embedding
            resolution: Leiden resolution parameter
            seed: Random seed for Leiden
            block_size: Number of query embeddings to search per FAISS batch

        Returns:
            Dictionary with clustering summary and assignments.
        """
        if not -1.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between -1.0 and 1.0")
        if max_neighbors < 1:
            raise ValueError("max_neighbors must be at least 1")
        if resolution <= 0:
            raise ValueError("resolution must be greater than 0")
        if block_size < 1:
            raise ValueError("block_size must be at least 1")

        embeddings = self.db.get_all_embeddings()
        total_nodes = len(self.db.chain_ids)

        if total_nodes == 0:
            return {
                "assignments": [],
                "summary": {
                    "db_path": str(self.db.db_path),
                    "index_name": self.db.index_name,
                    "threshold": threshold,
                    "max_neighbors": max_neighbors,
                    "resolution": resolution,
                    "seed": seed,
                    "total_nodes": 0,
                    "total_edges": 0,
                    "total_clusters": 0,
                    "singleton_clusters": 0,
                    "largest_cluster_size": 0,
                }
            }

        if self.use_gpu_for_search:
            self.db.move_to_gpu()

        edges, weights = self._build_similarity_graph(
            embeddings=embeddings,
            threshold=threshold,
            max_neighbors=max_neighbors,
            block_size=block_size
        )

        if edges:
            membership = self._run_leiden(
                total_nodes=total_nodes,
                edges=edges,
                weights=weights,
                resolution=resolution,
                seed=seed
            )
        else:
            membership = list(range(total_nodes))

        assignments, cluster_sizes = self._build_assignments(membership)
        summary = {
            "db_path": str(self.db.db_path),
            "index_name": self.db.index_name,
            "threshold": threshold,
            "max_neighbors": min(max_neighbors, max(total_nodes - 1, 0)),
            "resolution": resolution,
            "seed": seed,
            "total_nodes": total_nodes,
            "total_edges": len(edges),
            "total_clusters": len(cluster_sizes),
            "singleton_clusters": sum(1 for size in cluster_sizes.values() if size == 1),
            "largest_cluster_size": max(cluster_sizes.values(), default=0),
        }

        return {
            "assignments": assignments,
            "summary": summary
        }

    def _build_similarity_graph(
            self,
            embeddings,
            threshold: float,
            max_neighbors: int,
            block_size: int
    ) -> Tuple[List[Tuple[int, int]], List[float]]:
        """Build an approximate similarity graph using FAISS kNN search."""
        total_nodes = embeddings.shape[0]
        top_k = min(max_neighbors + 1, total_nodes)
        edge_weights: Dict[Tuple[int, int], float] = {}

        for start_idx in range(0, total_nodes, block_size):
            block = embeddings[start_idx:start_idx + block_size]
            scores, indices = self.db.index.search(block, top_k)

            for row_offset, (row_scores, row_indices) in enumerate(zip(scores.tolist(), indices.tolist())):
                source_idx = start_idx + row_offset
                for score, target_idx in zip(row_scores, row_indices):
                    if target_idx < 0 or target_idx == source_idx or score < threshold:
                        continue

                    edge = (source_idx, target_idx)
                    if source_idx > target_idx:
                        edge = (target_idx, source_idx)

                    previous = edge_weights.get(edge)
                    if previous is None or score > previous:
                        edge_weights[edge] = float(score)

        ordered_edges = sorted(edge_weights)
        ordered_weights = [edge_weights[edge] for edge in ordered_edges]
        return ordered_edges, ordered_weights

    def _run_leiden(
            self,
            total_nodes: int,
            edges: List[Tuple[int, int]],
            weights: List[float],
            resolution: float,
            seed: int
    ) -> List[int]:
        """Run Leiden on a weighted similarity graph."""
        import igraph
        import leidenalg

        graph = igraph.Graph(n=total_nodes, edges=edges, directed=False)
        graph.vs["name"] = self.db.chain_ids
        if weights:
            graph.es["weight"] = weights

        partition = leidenalg.find_partition(
            graph,
            leidenalg.RBConfigurationVertexPartition,
            weights=weights if weights else None,
            resolution_parameter=resolution,
            seed=seed
        )
        return list(partition.membership)

    def _build_assignments(self, membership: List[int]) -> Tuple[List[Dict[str, int | str]], Dict[str, int]]:
        """Convert Leiden membership labels into stable cluster IDs."""
        cluster_members: Dict[int, List[str]] = defaultdict(list)
        for chain_id, cluster_label in zip(self.db.chain_ids, membership):
            cluster_members[cluster_label].append(chain_id)

        ordered_clusters = sorted(
            cluster_members.values(),
            key=lambda members: (-len(members), min(members))
        )

        assignments = []
        cluster_sizes = {}
        for cluster_number, members in enumerate(ordered_clusters, 1):
            cluster_id = f"C{cluster_number:04d}"
            cluster_size = len(members)
            cluster_sizes[cluster_id] = cluster_size
            for chain_id in sorted(members):
                assignments.append({
                    "chain_id": chain_id,
                    "cluster_id": cluster_id,
                    "cluster_size": cluster_size
                })

        return assignments, cluster_sizes

    def print_summary(self, cluster_results: Dict):
        """Print a concise clustering summary."""
        summary = cluster_results["summary"]
        assignments = cluster_results["assignments"]

        print("\n" + "=" * 80)
        print("DATABASE CLUSTERING")
        print("=" * 80)
        print(f"Database path:        {summary['db_path']}")
        print(f"Index name:           {summary['index_name']}")
        print(f"Similarity threshold: {summary['threshold']}")
        print(f"Max neighbors:        {summary['max_neighbors']}")
        print(f"Resolution:           {summary['resolution']}")
        print(f"Total nodes:          {summary['total_nodes']}")
        print(f"Total edges:          {summary['total_edges']}")
        print(f"Total clusters:       {summary['total_clusters']}")
        print(f"Singleton clusters:   {summary['singleton_clusters']}")
        print(f"Largest cluster:      {summary['largest_cluster_size']}")
        print("=" * 80)

        if assignments:
            print(f"{'Cluster ID':<12} {'Size':<8} {'Representative':<40}")
            print("-" * 80)

            seen_clusters = set()
            for assignment in assignments:
                cluster_id = assignment["cluster_id"]
                if cluster_id in seen_clusters:
                    continue
                seen_clusters.add(cluster_id)
                print(
                    f"{cluster_id:<12} "
                    f"{assignment['cluster_size']:<8} "
                    f"{assignment['chain_id']:<40}"
                )

        print("=" * 80 + "\n")

    def export_results(
            self,
            cluster_results: Dict,
            output_file: str
    ):
        """Export cluster assignments to CSV."""
        with open(output_file, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Chain ID", "Cluster ID", "Cluster Size"])

            for assignment in cluster_results["assignments"]:
                writer.writerow([
                    assignment["chain_id"],
                    assignment["cluster_id"],
                    assignment["cluster_size"]
                ])

        print(f"Cluster assignments exported to {output_file}")
