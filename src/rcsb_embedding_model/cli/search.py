import os
import logging
import torch
import typer
from pathlib import Path
from typing import Annotated, Optional

from rcsb_embedding_model import __version__
from rcsb_embedding_model.types.api_types import StructureFormat
from rcsb_embedding_model.search.database_builder import EmbeddingDatabaseBuilder
from rcsb_embedding_model.search.structure_search import StructureSearch
from rcsb_embedding_model.search.clustering import EmbeddingClusterer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

app = typer.Typer(
    add_completion=False,
    help="3D structure search using embeddings and FAISS"
)


@app.command(
    name="build-db",
    help="Build an embedding database from a directory of structure files"
)
def build_database(
        structure_dir: Annotated[str, typer.Option(
            help='Directory containing structure files'
        )],
        output_db: Annotated[str, typer.Option(
            help='Path to save the FAISS database'
        )],
        structure_format: Annotated[StructureFormat, typer.Option(
            help='Structure file format (mmcif, binarycif, or pdb)'
        )] = StructureFormat.mmcif,
        file_extension: Annotated[Optional[str], typer.Option(
            help='File extension to filter (e.g., .cif, .bcif, or .pdb). If not specified, uses default for format'
        )] = None,
        min_res: Annotated[int, typer.Option(
            help='Minimum residue length for chains'
        )] = 10,
        max_res: Annotated[Optional[int], typer.Option(
            help='Maximum residue length for structures (None for no limit)'
        )] = None,
        device: Annotated[str, typer.Option(
            help='Device to use for embeddings (cuda, cpu, or auto)'
        )] = "auto",
        use_gpu_index: Annotated[bool, typer.Option(
            help='Use GPU for FAISS index (requires faiss-gpu)'
        )] = False
):
    """Build an embedding database from structure files."""

    # Parse output_db into directory and prefix
    # Files will be saved as: {output_db}.index and {output_db}.metadata
    output_db_path = Path(output_db)
    db_dir = output_db_path.parent
    index_name = output_db_path.name

    # Ensure we have a valid directory and prefix
    if not index_name:
        index_name = "embeddings"
    if db_dir == Path('.'):
        db_dir = Path.cwd()  # Use current directory explicitly

    # Determine device
    if device == "auto":
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        torch_device = torch.device(device)

    print(f"Using device for embeddings: {torch_device}")
    if use_gpu_index:
        print("GPU acceleration for FAISS index: enabled")

    builder = EmbeddingDatabaseBuilder(
        structure_dir=structure_dir,
        structure_format=structure_format,
        min_res=min_res,
        max_res=max_res,
        device=torch_device
    )

    builder.build_faiss_database(
        output_db=output_db,
        file_extension=file_extension,
        use_gpu_index=use_gpu_index
    )

    print(f"\nYou can now search this database using:")
    print(f"  search query --db-path {output_db} --query-structure <path_to_structure>")


@app.command(
    name="query",
    help="Search the database for similar structures"
)
def query_database(
        db_path: Annotated[str, typer.Option(
            help='Path to the FAISS database'
        )],
        query_structure: Annotated[str, typer.Option(
            help='Path to query structure file'
        )],
        structure_format: Annotated[StructureFormat, typer.Option(
            help='Structure file format (mmcif or pdb)'
        )] = StructureFormat.mmcif,
        chain_id: Annotated[Optional[str], typer.Option(
            help='Specific chain to search (if not specified, searches all chains)'
        )] = None,
        top_k: Annotated[int, typer.Option(
            help='Number of top results to return per chain'
        )] = 100,
        threshold: Annotated[Optional[float], typer.Option(
            help='Similarity score threshold to filter results (only return matches with score >= threshold)'
        )] = 0.8,
        output_csv: Annotated[Optional[str], typer.Option(
            help='Path to save results as CSV file (optional)'
        )] = None,
        min_res: Annotated[int, typer.Option(
            help='Minimum residue length for chains'
        )] = 10,
        max_res: Annotated[Optional[int], typer.Option(
            help='Maximum residue length for structures (None for no limit)'
        )] = None,
        device: Annotated[str, typer.Option(
            help='Device to use for embeddings (cuda, cpu, or auto)'
        )] = "auto",
        use_gpu_index: Annotated[bool, typer.Option(
            help='Use GPU for FAISS search (requires faiss-gpu)'
        )] = False
):
    """Search database for similar structures."""

    # Parse db_path into directory and index name
    db_dir, index_name = _parse_database_path(db_path)

    # Determine device
    if device == "auto":
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        torch_device = torch.device(device)

    print(f"Using device for embeddings: {torch_device}")
    if use_gpu_index:
        print("GPU acceleration for FAISS search: enabled")

    # Initialize search
    print("\nLoading database...")
    searcher = StructureSearch(
        db_path=str(db_dir),
        index_name=index_name,
        min_res=min_res,
        max_res=max_res,
        device=torch_device,
        use_gpu_for_search=use_gpu_index
    )

    # Display database statistics
    stats = searcher.get_db_statistics()
    print(f"Database contains {stats['total_chains']} chains")

    # Perform search
    print("\nPerforming search...")
    results = searcher.search_by_structure(
        query_structure=query_structure,
        structure_format=structure_format,
        chain_id=chain_id,
        top_k=top_k
    )

    # Filter by threshold if specified
    results = _filter_results_by_threshold(results, threshold)

    # Display results
    searcher.print_results(results)

    # Export if requested
    if output_csv:
        searcher.export_results(results, output_csv)


@app.command(
    name="query-db",
    help="Compare entries from a query database with a subject database"
)
def query_database_with_database(
        query_db_path: Annotated[str, typer.Option(
            help='Path to the query FAISS database'
        )],
        subject_db_path: Annotated[str, typer.Option(
            help='Path to the subject FAISS database to search against'
        )],
        top_k: Annotated[int, typer.Option(
            help='Number of top results to return per query chain'
        )] = 100,
        threshold: Annotated[Optional[float], typer.Option(
            help='Similarity score threshold to filter results (only return matches with score >= threshold)'
        )] = 0.8,
        output_csv: Annotated[Optional[str], typer.Option(
            help='Path to save results as CSV file (optional)'
        )] = None,
        use_gpu_index: Annotated[bool, typer.Option(
            help='Use GPU for FAISS search (requires faiss-gpu)'
        )] = False
):
    """Search subject database using all entries from query database."""

    # Parse subject_db_path
    query_db_dir, query_index_name = _parse_database_path(query_db_path)
    subject_db_dir, subject_index_name = _parse_database_path(subject_db_path)

    if use_gpu_index:
        print("GPU acceleration for FAISS search: enabled")

    # Load subject database
    print("\nLoading subject database...")
    searcher = StructureSearch(
        db_path=str(subject_db_dir),
        index_name=subject_index_name,
        use_gpu_for_search=use_gpu_index
    )

    # Display database statistics
    subject_stats = searcher.get_db_statistics()
    print(f"Subject database contains {subject_stats['total_chains']} chains")

    # Perform database-to-database search
    print("Performing search...")
    results = searcher.search_by_database(
        query_db_path=str(query_db_dir),
        query_index_name=query_index_name,
        top_k=top_k
    )

    # Filter by threshold if specified
    results = _filter_results_by_threshold(results, threshold)

    # Display results
    searcher.print_results(results)

    # Export if requested
    if output_csv:
        searcher.export_results(results, output_csv)


@app.command(
    name="stats",
    help="Display database statistics"
)
def show_statistics(
        db_path: Annotated[str, typer.Option(
            help='Path to the FAISS database (directory + prefix, e.g., ./db/my_db)'
        )]
):
    """Display database statistics."""
    # Parse db_path into directory and index name
    db_dir, index_name = _parse_database_path(db_path)
    searcher = StructureSearch(db_path=str(db_dir), index_name=index_name)
    stats = searcher.get_db_statistics()

    print("\n" + "="*80)
    print("DATABASE STATISTICS")
    print("="*80)
    print(f"Database path:    {stats['db_path']}")
    print(f"Index name:       {stats['index_name']}")
    print(f"Index type:       {stats['index_type']}")
    print(f"Dimension:        {stats['dimension']}")
    print(f"Total chains:     {stats['total_chains']}")
    print(f"On GPU:           {stats['on_gpu']}")
    print(f"GPU available:    {stats['gpu_available']}")
    print("="*80 + "\n")


@app.command(
    name="cluster",
    help="Cluster database embeddings using Leiden algorithm on similarity graph"
)
def cluster_database(
        db_path: Annotated[str, typer.Option(
            help='Path to the FAISS database'
        )],
        threshold: Annotated[float, typer.Option(
            help='Similarity threshold for edge creation (0-1, where 1.0 = identical)'
        )] = 0.8,
        resolution: Annotated[float, typer.Option(
            help='Leiden resolution parameter (higher = more clusters)'
        )] = 1.0,
        output: Annotated[str, typer.Option(
            help='Path to save cluster assignments (CSV or JSON)'
        )] = "clusters.csv",
        max_neighbors: Annotated[int, typer.Option(
            help='Maximum number of neighbors to consider per chain'
        )] = 1000,
        min_cluster_size: Annotated[Optional[int], typer.Option(
            help='Minimum cluster size to include in output (filters smaller clusters)'
        )] = None,
        use_gpu_index: Annotated[bool, typer.Option(
            help='Use GPU for FAISS operations (requires faiss-gpu)'
        )] = False,
        seed: Annotated[Optional[int], typer.Option(
            help='Random seed for reproducibility'
        )] = None
):
    """Cluster database embeddings using Leiden algorithm."""

    # Parse db_path into directory and index name
    db_dir, index_name = _parse_database_path(db_path)

    if use_gpu_index:
        print("GPU acceleration for FAISS operations: enabled")

    # Initialize clusterer
    print("Initializing clusterer...")
    clusterer = EmbeddingClusterer(db_path=str(db_dir), index_name=index_name)
    clusterer.load_database(use_gpu=use_gpu_index)

    # Build similarity graph
    clusterer.build_similarity_graph(
        threshold=threshold,
        max_neighbors=max_neighbors
    )

    # Perform clustering
    clusterer.cluster_leiden(
        resolution=resolution,
        seed=seed
    )

    # Display statistics
    clusterer.print_statistics()

    # Determine output format from file extension
    output_path = Path(output)
    if output_path.suffix == '.json':
        output_format = 'json'
    else:
        output_format = 'csv'

    # Export results
    clusterer.export_clusters(
        output_file=output,
        format=output_format,
        min_cluster_size=min_cluster_size
    )


def version_callback(value: bool):
    if value:
        typer.echo(f"{__version__}")
        raise typer.Exit()


@app.callback()
def main(
        version: bool = typer.Option(
            None,
            "--version",
            callback=version_callback,
            is_eager=True,
            help="Show the version and exit",
        )
):
    pass

def _parse_database_path(db_path: str, default_index_name: str = "structure_embeddings") -> tuple[Path, str]:
    """Split a database path into its directory and index name components."""
    db_path_obj = Path(db_path)
    db_dir = db_path_obj.parent
    index_name = db_path_obj.name

    if not index_name:
        index_name = default_index_name
    if db_dir == Path('.'):
        db_dir = Path.cwd()

    return db_dir, index_name


def _filter_results_by_threshold(results, threshold: float | None):
    """Filter search results by similarity threshold."""
    if threshold is None:
        return results

    print(f"Filtering results with similarity score threshold >= {threshold}")
    filtered_results = {}
    total_before = sum(len(ids) for ids, _ in results.values())
    for query_chain, (chain_ids, scores) in results.items():
        filtered_pairs = [(cid, score) for cid, score in zip(chain_ids, scores) if score >= threshold]
        if filtered_pairs:
            filtered_chain_ids, filtered_scores = zip(*filtered_pairs)
            filtered_results[query_chain] = (list(filtered_chain_ids), list(filtered_scores))
        else:
            filtered_results[query_chain] = ([], [])
    total_after = sum(len(ids) for ids, _ in filtered_results.values())
    print(f"Filtered from {total_before} to {total_after} results")
    return filtered_results


if __name__ == "__main__":
    app()
