import os
import torch
import typer
from typing import Annotated, Optional

from rcsb_embedding_model import __version__
from rcsb_embedding_model.types.api_types import StructureFormat
from rcsb_embedding_model.search.database_builder import EmbeddingDatabaseBuilder
from rcsb_embedding_model.search.chroma_database import ChromaEmbeddingDatabase
from rcsb_embedding_model.search.structure_search import StructureSearch

os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = typer.Typer(
    add_completion=False,
    help="3D structure search using embeddings and ChromaDB"
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
            help='Path to save the ChromaDB database directory'
        )],
        temp_file: Annotated[str, typer.Option(
            help='Temporary file to save intermediate embeddings (torch format)'
        )] = "temp_embeddings.pt",
        structure_format: Annotated[StructureFormat, typer.Option(
            help='Structure file format (mmcif or pdb)'
        )] = StructureFormat.mmcif,
        file_extension: Annotated[Optional[str], typer.Option(
            help='File extension to filter (e.g., .cif, .pdb). If not specified, uses default for format'
        )] = None,
        collection_name: Annotated[str, typer.Option(
            help='Name of the ChromaDB collection'
        )] = "structure_embeddings",
        min_res: Annotated[int, typer.Option(
            help='Minimum residue length for chains'
        )] = 10,
        max_res: Annotated[Optional[int], typer.Option(
            help='Maximum residue length for structures (None for no limit)'
        )] = None,
        device: Annotated[str, typer.Option(
            help='Device to use (cuda, cpu, or auto)'
        )] = "auto"
):
    """Build an embedding database from structure files."""

    # Determine device
    if device == "auto":
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        torch_device = torch.device(device)

    print(f"Using device: {torch_device}")

    # Step 1: Build embeddings from structure files
    print("\n" + "="*80)
    print("STEP 1: Building embeddings from structure files")
    print("="*80 + "\n")

    builder = EmbeddingDatabaseBuilder(
        structure_dir=structure_dir,
        structure_format=structure_format,
        min_res=min_res,
        max_res=max_res,
        device=torch_device
    )

    chain_ids, embeddings = builder.build_database(
        output_path=temp_file,
        file_extension=file_extension
    )

    # Step 2: Create ChromaDB database
    print("\n" + "="*80)
    print("STEP 2: Creating ChromaDB database")
    print("="*80 + "\n")

    db = ChromaEmbeddingDatabase(db_path=output_db, collection_name=collection_name)
    db.create_database(chain_ids=chain_ids, embeddings=embeddings)

    print("\n" + "="*80)
    print("Database build complete!")
    print("="*80)
    print(f"Database location: {output_db}")
    print(f"Collection name: {collection_name}")
    print(f"Total chains: {len(chain_ids)}")
    print(f"\nYou can now search this database using:")
    print(f"  search query --db-path {output_db} --query-structure <path_to_structure>")


@app.command(
    name="query",
    help="Search the database for similar structures"
)
def query_database(
        db_path: Annotated[str, typer.Option(
            help='Path to the ChromaDB database directory'
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
        collection_name: Annotated[str, typer.Option(
            help='Name of the ChromaDB collection'
        )] = "structure_embeddings",
        top_k: Annotated[int, typer.Option(
            help='Number of top results to return per chain'
        )] = 10,
        threshold: Annotated[Optional[float], typer.Option(
            help='Distance threshold to filter results (only return matches with distance <= threshold)'
        )] = None,
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
            help='Device to use (cuda, cpu, or auto)'
        )] = "auto"
):
    """Search database for similar structures."""

    # Determine device
    if device == "auto":
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        torch_device = torch.device(device)

    print(f"Using device: {torch_device}")

    # Initialize search
    print("\nLoading database...")
    searcher = StructureSearch(
        db_path=db_path,
        collection_name=collection_name,
        min_res=min_res,
        max_res=max_res,
        device=torch_device
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
    if threshold is not None:
        print(f"Filtering results with distance threshold <= {threshold}")
        filtered_results = {}
        total_before = sum(len(ids) for ids, _ in results.values())
        for query_chain, (chain_ids, distances) in results.items():
            filtered_pairs = [(cid, dist) for cid, dist in zip(chain_ids, distances) if dist <= threshold]
            if filtered_pairs:
                filtered_chain_ids, filtered_distances = zip(*filtered_pairs)
                filtered_results[query_chain] = (list(filtered_chain_ids), list(filtered_distances))
            else:
                filtered_results[query_chain] = ([], [])
        total_after = sum(len(ids) for ids, _ in filtered_results.values())
        print(f"Filtered from {total_before} to {total_after} results")
        results = filtered_results

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
            help='Path to the ChromaDB database directory'
        )],
        collection_name: Annotated[str, typer.Option(
            help='Name of the ChromaDB collection'
        )] = "structure_embeddings"
):
    """Display database statistics."""
    db = ChromaEmbeddingDatabase(db_path=db_path, collection_name=collection_name)
    db.load_database()

    stats = db.get_statistics()

    print("\n" + "="*80)
    print("DATABASE STATISTICS")
    print("="*80)
    print(f"Database path:    {stats['db_path']}")
    print(f"Collection name:  {stats['collection_name']}")
    print(f"Total chains:     {stats['total_chains']}")
    print("="*80 + "\n")


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


if __name__ == "__main__":
    app()
