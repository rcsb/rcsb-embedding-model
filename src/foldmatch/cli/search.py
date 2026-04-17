import os
import logging
import torch
import pandas as pd
import typer
from pathlib import Path
from typing import Annotated, Optional, List

from foldmatch import __version__
from foldmatch.cli.args_utils import arg_devices, set_log_level
from foldmatch.types.api_types import StructureFormat, Accelerator, Strategy, Granularity, LogLevel
from foldmatch.search.database_builder import EmbeddingDatabaseBuilder
from foldmatch.search.faiss_database import FaissEmbeddingDatabase
from foldmatch.search.structure_search import StructureSearch
from foldmatch.search.clustering import EmbeddingClusterer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = typer.Typer(
    add_completion=False,
    help=f"3D structure search using embeddings and FAISS. Version: {__version__}."
)

build_db_app = typer.Typer(
    add_completion=False,
    help="Build an embedding database."
)
app.add_typer(build_db_app, name="build-db")

update_db_app = typer.Typer(
    add_completion=False,
    help="Update an existing embedding database."
)
app.add_typer(update_db_app, name="update-db")


@build_db_app.command(
    name="structures",
    help="Build an embedding database from a directory of structure files."
)
def build_database_from_structures(
        structure_dir: Annotated[str, typer.Option(
            help='Directory containing structure files.'
        )],
        output_db: Annotated[str, typer.Option(
            help='Path to save the FAISS database.'
        )],
        tmp_dir: Annotated[str, typer.Option(
            help='Temporal directory.'
        )],
        structure_format: Annotated[StructureFormat, typer.Option(
            help='Structure file format (mmcif, binarycif, or pdb).'
        )] = StructureFormat.mmcif,
        granularity: Annotated[Granularity, typer.Option(
            help='Calculate embeddings for "chain" or "assembly" level.'
        )] = 'chain',
        file_extension: Annotated[Optional[str], typer.Option(
            help='File extension to filter (e.g., .cif, .bcif, or .pdb). If not specified, uses default for format.'
        )] = None,
        min_res: Annotated[int, typer.Option(
            help='Minimum residue length for chains.'
        )] = 10,
        use_gpu_index: Annotated[bool, typer.Option(
            help='Use GPU for FAISS index (requires faiss-gpu).'
        )] = False,
        accelerator: Annotated[Accelerator, typer.Option(
            help='Device used for inference.'
        )] = "auto",
        devices: Annotated[List[str], typer.Option(
            help='The devices to use. Can be set to a positive number or "auto". Repeat this argument to indicate multiple indices of devices. "auto" for automatic selection based on the chosen accelerator.'
        )] = tuple(['auto']),
        strategy: Annotated[Strategy, typer.Option(
            help='Lightning strategy to control distribution of inference.'
        )] = 'auto',
        batch_size_res: Annotated[int, typer.Option(
            help='Number of samples processed together in one iteration.'
        )] = 1,
        num_workers_res: Annotated[int, typer.Option(
            help='Number of subprocesses to use for data loading.'
        )] = 0,
        num_nodes_res: Annotated[int, typer.Option(
            help='Number of nodes to use for inference of residue embeddings.'
        )] = 1,
        batch_size_aggregator: Annotated[int, typer.Option(
            help='Number of samples processed together in one iteration.'
        )] = 1,
        num_workers_aggregator: Annotated[int, typer.Option(
            help='Number of subprocesses to use for data loading.'
        )] = 0,
        num_nodes_aggregator: Annotated[int, typer.Option(
            help='Number of nodes to use for inference of embeddings.'
        )] = 1,
        log_level: Annotated[LogLevel, typer.Option(
            help='Number of nodes to use for inference of embeddings.'
        )] = 'info'
):
    """Build an embedding database from structure files."""

    set_log_level(log_level)

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
    output_db = str(db_dir / index_name)

    logging.info(f"Using device for embeddings: {str(accelerator.value) if hasattr(accelerator, 'value') else accelerator}")
    if use_gpu_index:
        logging.info("GPU acceleration for FAISS index: enabled")

    builder = EmbeddingDatabaseBuilder(
        structure_dir=structure_dir,
        tmp_dir=tmp_dir,
        structure_format=structure_format,
        min_res=min_res,
        accelerator=accelerator
    )

    builder.build_faiss_database(
        output_db=output_db,
        granularity=granularity,
        devices=arg_devices(devices),
        strategy=strategy,
        file_extension=file_extension,
        use_gpu_index=use_gpu_index,
        batch_size_res=batch_size_res,
        num_workers_res=num_workers_res,
        num_nodes_res=num_nodes_res,
        batch_size_chain=batch_size_aggregator,
        num_workers_chain=num_workers_aggregator,
        num_nodes_chain=num_nodes_aggregator
    )
    import torch.distributed as dist
    if not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0:
        logging.info(f"You can now search this database using:")
        logging.info(f"   fm-search query --db-path {output_db} --query-structure <path_to_structure>")


@update_db_app.command(
    name="structures",
    help="Update an existing embedding database with new or replacement structure files."
)
def update_database_from_structures(
        structure_dir: Annotated[str, typer.Option(
            help='Directory containing new or updated structure files.'
        )],
        output_db: Annotated[str, typer.Option(
            help='Path to the existing FAISS database to update.'
        )],
        tmp_dir: Annotated[str, typer.Option(
            help='Temporal directory.'
        )],
        structure_format: Annotated[StructureFormat, typer.Option(
            help='Structure file format (mmcif, binarycif, or pdb).'
        )] = StructureFormat.mmcif,
        granularity: Annotated[Granularity, typer.Option(
            help='Calculate embeddings for "chain" or "assembly" level.'
        )] = 'chain',
        file_extension: Annotated[Optional[str], typer.Option(
            help='File extension to filter (e.g., .cif, .bcif, or .pdb). If not specified, uses default for format.'
        )] = None,
        min_res: Annotated[int, typer.Option(
            help='Minimum residue length for chains.'
        )] = 10,
        use_gpu_index: Annotated[bool, typer.Option(
            help='Use GPU for FAISS index (requires faiss-gpu).'
        )] = False,
        accelerator: Annotated[Accelerator, typer.Option(
            help='Device used for inference.'
        )] = "auto",
        devices: Annotated[List[str], typer.Option(
            help='The devices to use. Can be set to a positive number or "auto". Repeat this argument to indicate multiple indices of devices. "auto" for automatic selection based on the chosen accelerator.'
        )] = tuple(['auto']),
        strategy: Annotated[Strategy, typer.Option(
            help='Lightning strategy to control distribution of inference.'
        )] = 'auto',
        batch_size_res: Annotated[int, typer.Option(
            help='Number of samples processed together in one iteration.'
        )] = 1,
        num_workers_res: Annotated[int, typer.Option(
            help='Number of subprocesses to use for data loading.'
        )] = 0,
        num_nodes_res: Annotated[int, typer.Option(
            help='Number of nodes to use for inference of residue embeddings.'
        )] = 1,
        batch_size_aggregator: Annotated[int, typer.Option(
            help='Number of samples processed together in one iteration.'
        )] = 1,
        num_workers_aggregator: Annotated[int, typer.Option(
            help='Number of subprocesses to use for data loading.'
        )] = 0,
        num_nodes_aggregator: Annotated[int, typer.Option(
            help='Number of nodes to use for inference of embeddings.'
        )] = 1,
        log_level: Annotated[LogLevel, typer.Option(
            help='Number of nodes to use for inference of embeddings.'
        )] = 'info'
):
    """Update an existing embedding database with new or replacement structure files."""

    set_log_level(log_level)

    # Parse output_db into directory and prefix
    output_db_path = Path(output_db)
    db_dir = output_db_path.parent
    index_name = output_db_path.name

    if not index_name:
        index_name = "embeddings"
    if db_dir == Path('.'):
        db_dir = Path.cwd()
    output_db = str(db_dir / index_name)

    logging.info(f"Using device for embeddings: {str(accelerator.value) if hasattr(accelerator, 'value') else accelerator}")
    if use_gpu_index:
        logging.info("GPU acceleration for FAISS index: enabled")

    builder = EmbeddingDatabaseBuilder(
        structure_dir=structure_dir,
        tmp_dir=tmp_dir,
        structure_format=structure_format,
        min_res=min_res,
        accelerator=accelerator
    )

    builder.update_faiss_database(
        output_db=output_db,
        granularity=granularity,
        devices=arg_devices(devices),
        strategy=strategy,
        file_extension=file_extension,
        use_gpu_index=use_gpu_index,
        batch_size_res=batch_size_res,
        num_workers_res=num_workers_res,
        num_nodes_res=num_nodes_res,
        batch_size_chain=batch_size_aggregator,
        num_workers_chain=num_workers_aggregator,
        num_nodes_chain=num_nodes_aggregator
    )
    import torch.distributed as dist
    if not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0:
        logging.info(f"Database updated successfully: {output_db}")


@build_db_app.command(
    name="embeddings",
    help="Build an embedding database from a directory of pre-computed embedding files (.csv or .pt)."
)
def build_database_from_embeddings(
        embedding_dir: Annotated[str, typer.Option(
            help='Directory containing pre-computed embedding files (.csv or .pt).'
        )],
        output_db: Annotated[str, typer.Option(
            help='Path to save the FAISS database.'
        )],
        file_extension: Annotated[Optional[str], typer.Option(
            help='File extension to filter (e.g., .csv or .pt). If not specified, collects both.'
        )] = None,
        use_gpu_index: Annotated[bool, typer.Option(
            help='Use GPU for FAISS index (requires faiss-gpu).'
        )] = False,
        log_level: Annotated[LogLevel, typer.Option(
            help='Logging level.'
        )] = 'info'
):
    """Build an embedding database from pre-computed embedding files."""

    set_log_level(log_level)

    db_dir, index_name, output_db = _parse_output_db(output_db)
    chain_ids, embeddings = _load_embeddings_from_dir(embedding_dir, file_extension)

    logging.info(f"Loaded {len(embeddings)} embeddings from {embedding_dir}")

    db = FaissEmbeddingDatabase(db_path=str(db_dir), index_name=index_name)
    db.create_database(chain_ids=chain_ids, embeddings=embeddings, use_gpu=use_gpu_index)

    logging.info(f"Database created: {output_db}")
    logging.info(f"Total embeddings: {len(chain_ids)}")


@update_db_app.command(
    name="embeddings",
    help="Update an existing embedding database with new or replacement embeddings from pre-computed files (.csv or .pt)."
)
def update_database_from_embeddings(
        embedding_dir: Annotated[str, typer.Option(
            help='Directory containing pre-computed embedding files (.csv or .pt).'
        )],
        output_db: Annotated[str, typer.Option(
            help='Path to the existing FAISS database to update.'
        )],
        file_extension: Annotated[Optional[str], typer.Option(
            help='File extension to filter (e.g., .csv or .pt). If not specified, collects both.'
        )] = None,
        use_gpu_index: Annotated[bool, typer.Option(
            help='Use GPU for FAISS index (requires faiss-gpu).'
        )] = False,
        log_level: Annotated[LogLevel, typer.Option(
            help='Logging level.'
        )] = 'info'
):
    """Update an existing embedding database with new or replacement embeddings from pre-computed files."""

    set_log_level(log_level)

    db_dir, index_name, output_db = _parse_output_db(output_db)
    chain_ids, embeddings = _load_embeddings_from_dir(embedding_dir, file_extension)

    logging.info(f"Loaded {len(embeddings)} embeddings from {embedding_dir}")

    db = FaissEmbeddingDatabase(db_path=str(db_dir), index_name=index_name)
    db.load_database()

    logging.info(f"Existing database contains {len(db.chain_ids)} embeddings")

    db.update_embeddings(chain_ids=chain_ids, embeddings=embeddings, use_gpu=use_gpu_index)

    logging.info(f"Database updated: {output_db}")
    logging.info(f"Total embeddings: {len(db.chain_ids)}")


@build_db_app.command(
    name="fasta",
    help="Build an embedding database from protein sequences in a FASTA file."
)
def build_database_from_fasta(
        fasta_file: Annotated[str, typer.Option(
            help='FASTA file containing protein sequences.'
        )],
        output_db: Annotated[str, typer.Option(
            help='Path to save the FAISS database.'
        )],
        tmp_dir: Annotated[str, typer.Option(
            help='Directory for intermediate residue embeddings.'
        )],
        min_res_n: Annotated[int, typer.Option(
            help='Consider only sequences with at least <min_res_n> residues.'
        )] = 0,
        use_gpu_index: Annotated[bool, typer.Option(
            help='Use GPU for FAISS index (requires faiss-gpu).'
        )] = False,
        accelerator: Annotated[Accelerator, typer.Option(
            help='Device used for inference.'
        )] = "auto",
        devices: Annotated[List[str], typer.Option(
            help='The devices to use. Can be set to a positive number or "auto".'
        )] = tuple(['auto']),
        strategy: Annotated[Strategy, typer.Option(
            help='Lightning strategy to control distribution of inference.'
        )] = 'auto',
        batch_size: Annotated[int, typer.Option(
            help='Number of samples processed together for residue embedding inference.'
        )] = 1,
        num_workers: Annotated[int, typer.Option(
            help='Number of subprocesses to use for data loading.'
        )] = 0,
        num_nodes: Annotated[int, typer.Option(
            help='Number of nodes to use for residue embedding inference.'
        )] = 1,
        batch_size_aggregator: Annotated[int, typer.Option(
            help='Number of samples processed together for chain embedding inference.'
        )] = 1,
        num_workers_aggregator: Annotated[int, typer.Option(
            help='Number of subprocesses to use for chain embedding data loading.'
        )] = 0,
        num_nodes_aggregator: Annotated[int, typer.Option(
            help='Number of nodes to use for chain embedding inference.'
        )] = 1,
        compute_residue_embedding: Annotated[bool, typer.Option(
            help='Compute residue level embeddings first. When disabled, pre-computed residue embeddings must exist in tmp-dir.'
        )] = True,
        log_level: Annotated[LogLevel, typer.Option(
            help='Logging level.'
        )] = 'info'
):
    """Build an embedding database from protein sequences in a FASTA file."""

    set_log_level(log_level)

    db_dir, index_name, output_db = _parse_output_db(output_db)
    chain_ids, embeddings = _compute_fasta_embeddings(
        fasta_file=fasta_file, tmp_dir=tmp_dir, min_res_n=min_res_n,
        compute_residue_embedding=compute_residue_embedding,
        accelerator=accelerator, devices=devices, strategy=strategy,
        batch_size=batch_size, num_workers=num_workers, num_nodes=num_nodes,
        batch_size_aggregator=batch_size_aggregator,
        num_workers_aggregator=num_workers_aggregator,
        num_nodes_aggregator=num_nodes_aggregator
    )

    db = FaissEmbeddingDatabase(db_path=str(db_dir), index_name=index_name)
    db.create_database(chain_ids=chain_ids, embeddings=embeddings, use_gpu=use_gpu_index)

    logging.info(f"Database created: {output_db}")
    logging.info(f"Total embeddings: {len(chain_ids)}")


@update_db_app.command(
    name="fasta",
    help="Update an existing embedding database with new or replacement embeddings from protein sequences in a FASTA file."
)
def update_database_from_fasta(
        fasta_file: Annotated[str, typer.Option(
            help='FASTA file containing protein sequences.'
        )],
        output_db: Annotated[str, typer.Option(
            help='Path to the existing FAISS database to update.'
        )],
        tmp_dir: Annotated[str, typer.Option(
            help='Directory for intermediate residue embeddings.'
        )],
        min_res_n: Annotated[int, typer.Option(
            help='Consider only sequences with at least <min_res_n> residues.'
        )] = 0,
        use_gpu_index: Annotated[bool, typer.Option(
            help='Use GPU for FAISS index (requires faiss-gpu).'
        )] = False,
        accelerator: Annotated[Accelerator, typer.Option(
            help='Device used for inference.'
        )] = "auto",
        devices: Annotated[List[str], typer.Option(
            help='The devices to use. Can be set to a positive number or "auto".'
        )] = tuple(['auto']),
        strategy: Annotated[Strategy, typer.Option(
            help='Lightning strategy to control distribution of inference.'
        )] = 'auto',
        batch_size: Annotated[int, typer.Option(
            help='Number of samples processed together for residue embedding inference.'
        )] = 1,
        num_workers: Annotated[int, typer.Option(
            help='Number of subprocesses to use for data loading.'
        )] = 0,
        num_nodes: Annotated[int, typer.Option(
            help='Number of nodes to use for residue embedding inference.'
        )] = 1,
        batch_size_aggregator: Annotated[int, typer.Option(
            help='Number of samples processed together for chain embedding inference.'
        )] = 1,
        num_workers_aggregator: Annotated[int, typer.Option(
            help='Number of subprocesses to use for chain embedding data loading.'
        )] = 0,
        num_nodes_aggregator: Annotated[int, typer.Option(
            help='Number of nodes to use for chain embedding inference.'
        )] = 1,
        compute_residue_embedding: Annotated[bool, typer.Option(
            help='Compute residue level embeddings first. When disabled, pre-computed residue embeddings must exist in tmp-dir.'
        )] = True,
        log_level: Annotated[LogLevel, typer.Option(
            help='Logging level.'
        )] = 'info'
):
    """Update an existing embedding database with new or replacement embeddings from FASTA sequences."""

    set_log_level(log_level)

    db_dir, index_name, output_db = _parse_output_db(output_db)
    chain_ids, embeddings = _compute_fasta_embeddings(
        fasta_file=fasta_file, tmp_dir=tmp_dir, min_res_n=min_res_n,
        compute_residue_embedding=compute_residue_embedding,
        accelerator=accelerator, devices=devices, strategy=strategy,
        batch_size=batch_size, num_workers=num_workers, num_nodes=num_nodes,
        batch_size_aggregator=batch_size_aggregator,
        num_workers_aggregator=num_workers_aggregator,
        num_nodes_aggregator=num_nodes_aggregator
    )

    db = FaissEmbeddingDatabase(db_path=str(db_dir), index_name=index_name)
    db.load_database()

    logging.info(f"Existing database contains {len(db.chain_ids)} embeddings")

    db.update_embeddings(chain_ids=chain_ids, embeddings=embeddings, use_gpu=use_gpu_index)

    logging.info(f"Database updated: {output_db}")
    logging.info(f"Total embeddings: {len(db.chain_ids)}")


@app.command(
    name="query",
    help="Search the database for similar structures."
)
def query_database(
        db_path: Annotated[str, typer.Option(
            help='Path to the FAISS database.'
        )],
        query_structure: Annotated[str, typer.Option(
            help='Path to query structure file.'
        )],
        structure_format: Annotated[StructureFormat, typer.Option(
            help='Structure file format (mmcif, binarycif or pdb)'
        )] = StructureFormat.mmcif,
        granularity: Annotated[Granularity, typer.Option(
            help='Query database for "chain" or "assembly" embeddings.'
        )] = 'chain',
        chain_id: Annotated[Optional[str], typer.Option(
            help='When "granularity=chain", specific chain to search (if not specified, searches all chains).'
        )] = None,
        assembly_id: Annotated[Optional[str], typer.Option(
            help='When "granularity=assembly", specific assembly to search (if not specified, searches by asymmetric unit).'
        )] = None,
        top_k: Annotated[int, typer.Option(
            help='Number of top results to return per chain.'
        )] = 100,
        threshold: Annotated[Optional[float], typer.Option(
            help='Similarity score threshold to filter results (only return matches with score >= threshold).'
        )] = 0.8,
        output_csv: Annotated[Optional[str], typer.Option(
            help='Path to save results as CSV file (optional).'
        )] = None,
        min_res: Annotated[int, typer.Option(
            help='Minimum residue length for chains.'
        )] = 10,
        max_res: Annotated[Optional[int], typer.Option(
            help='Maximum residue length for structures (None for no limit).'
        )] = None,
        device: Annotated[str, typer.Option(
            help='Device to use for embedding calculation (cuda, cpu, or auto).'
        )] = "auto",
        use_gpu_index: Annotated[bool, typer.Option(
            help='Use GPU for FAISS search (requires faiss-gpu).'
        )] = False,
        log_level: Annotated[LogLevel, typer.Option(
            help='Number of nodes to use for inference of embeddings.'
        )] = 'info'
):
    """Search database for similar structures."""

    set_log_level(log_level)

    # Parse db_path into directory and index name
    db_dir, index_name = _parse_database_path(db_path)

    # Determine device
    if device == "auto":
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        torch_device = torch.device(device)

    logging.info(f"Using device for embeddings DB: {str(torch_device)}")
    if use_gpu_index:
        logging.info("GPU acceleration for FAISS search: enabled")

    # Initialize search
    logging.info("Loading database...")
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

    # Perform search
    logging.info("Performing search...")
    results = searcher.search_by_structure(
        query_structure=query_structure,
        structure_format=structure_format,
        granularity=granularity,
        chain_id=chain_id,
        assembly_id=assembly_id,
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
    help="Compare entries from a query database with a subject database."
)
def query_database_with_database(
        query_db_path: Annotated[str, typer.Option(
            help='Path to the query FAISS database.'
        )],
        subject_db_path: Annotated[str, typer.Option(
            help='Path to the subject FAISS database to search against.'
        )],
        top_k: Annotated[int, typer.Option(
            help='Number of top results to return per query chain.'
        )] = 100,
        threshold: Annotated[Optional[float], typer.Option(
            help='Similarity score threshold to filter results (only return matches with score >= threshold).'
        )] = 0.8,
        output_csv: Annotated[Optional[str], typer.Option(
            help='Path to save results as CSV file (optional).'
        )] = None,
        use_gpu_index: Annotated[bool, typer.Option(
            help='Use GPU for FAISS search (requires faiss-gpu).'
        )] = False,
        log_level: Annotated[LogLevel, typer.Option(
            help='Number of nodes to use for inference of embeddings.'
        )] = 'info'
):
    """Search subject database using all entries from query database."""

    set_log_level(log_level)

    # Parse subject_db_path
    query_db_dir, query_index_name = _parse_database_path(query_db_path)
    subject_db_dir, subject_index_name = _parse_database_path(subject_db_path)

    if use_gpu_index:
        logging.info("GPU acceleration for FAISS search: enabled")

    # Load subject database
    logging.info("\nLoading subject database...")
    searcher = StructureSearch(
        db_path=str(subject_db_dir),
        index_name=subject_index_name,
        use_gpu_for_search=use_gpu_index
    )

    # Display database statistics
    subject_stats = searcher.get_db_statistics()
    logging.info(f"Subject database contains {subject_stats['total_embeddings']} chains")

    # Perform database-to-database search
    logging.info("Performing search...")
    results = searcher.search_by_database(
        query_db_path=str(query_db_dir),
        query_index_name=query_index_name,
        top_k=top_k
    )

    # Filter by threshold if specified
    results = _filter_results_by_threshold(results, threshold)

    # Display results only if not exporting to CSV
    if not output_csv:
        searcher.print_results(results)

    # Export if requested
    if output_csv:
        searcher.export_results(results, output_csv)


@app.command(
    name="stats",
    help="Display database statistics."
)
def show_statistics(
        db_path: Annotated[str, typer.Option(
            help='Path to the FAISS database (directory + prefix, e.g., ./db/my_db).'
        )],
        log_level: Annotated[LogLevel, typer.Option(
            help='Number of nodes to use for inference of embeddings.'
        )] = 'info'
):
    """Display database statistics."""

    set_log_level(log_level)

    # Parse db_path into directory and index name
    db_dir, index_name = _parse_database_path(db_path)
    searcher = StructureSearch(db_path=str(db_dir), index_name=index_name)
    stats = searcher.get_db_statistics()

    logging.info("DATABASE STATISTICS")
    logging.info("="*80)
    logging.info(f"Database path:    {stats['db_path']}")
    logging.info(f"Index name:       {stats['index_name']}")
    logging.info(f"Index type:       {stats['index_type']}")
    logging.info(f"Dimension:        {stats['dimension']}")
    logging.info(f"Total embeddings: {stats['total_embeddings']}")
    logging.info(f"On GPU:           {stats['on_gpu']}")
    logging.info(f"GPU available:    {stats['gpu_available']}")
    logging.info("="*80 + "\n")


@app.command(
    name="cluster",
    help="Cluster database embeddings using Leiden algorithm on similarity graph."
)
def cluster_database(
        db_path: Annotated[str, typer.Option(
            help='Path to the FAISS database.'
        )],
        threshold: Annotated[float, typer.Option(
            help='Similarity threshold for edge creation (0-1, where 1.0 = identical).'
        )] = 0.8,
        resolution: Annotated[float, typer.Option(
            help='Leiden resolution parameter (higher = more clusters).'
        )] = 1.0,
        output: Annotated[str, typer.Option(
            help='Path to save cluster assignments (CSV or JSON).'
        )] = "clusters.csv",
        max_neighbors: Annotated[int, typer.Option(
            help='Maximum number of neighbors to consider per chain'
        )] = 1000,
        min_cluster_size: Annotated[Optional[int], typer.Option(
            help='Minimum cluster size to include in output (filters smaller clusters).'
        )] = None,
        use_gpu_index: Annotated[bool, typer.Option(
            help='Use GPU for FAISS operations (requires faiss-gpu).'
        )] = False,
        seed: Annotated[Optional[int], typer.Option(
            help='Random seed for reproducibility.'
        )] = None,
        log_level: Annotated[LogLevel, typer.Option(
            help='Number of nodes to use for inference of embeddings.'
        )] = 'info'
):
    """Cluster database embeddings using Leiden algorithm."""

    set_log_level(log_level)

    # Parse db_path into directory and index name
    db_dir, index_name = _parse_database_path(db_path)

    if use_gpu_index:
        logging.info("GPU acceleration for FAISS operations: enabled")

    # Initialize clusterer
    logging.info("Initializing clusterer...")
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


@app.command(
    name="similarity-graph",
    help="Build a similarity graph from database embeddings and export it in GraphML format."
)
def similarity_graph(
        db_path: Annotated[str, typer.Option(
            help='Path to the FAISS database.'
        )],
        threshold: Annotated[float, typer.Option(
            help='Similarity threshold for edge creation (0-1, where 1.0 = identical).'
        )] = 0.8,
        output: Annotated[str, typer.Option(
            help='Path to save the similarity graph (GraphML format).'
        )] = "similarity_graph.graphml",
        max_neighbors: Annotated[int, typer.Option(
            help='Maximum number of neighbors to consider per chain.'
        )] = 1000,
        use_gpu_index: Annotated[bool, typer.Option(
            help='Use GPU for FAISS operations (requires faiss-gpu).'
        )] = False,
        log_level: Annotated[LogLevel, typer.Option(
            help='Logging verbosity level.'
        )] = 'info'
):
    """Build and export a similarity graph from database embeddings."""

    set_log_level(log_level)

    db_dir, index_name = _parse_database_path(db_path)

    if use_gpu_index:
        logging.info("GPU acceleration for FAISS operations: enabled")

    logging.info("Initializing clusterer...")
    clusterer = EmbeddingClusterer(db_path=str(db_dir), index_name=index_name)
    clusterer.load_database(use_gpu=use_gpu_index)

    graph = clusterer.build_similarity_graph(
        threshold=threshold,
        max_neighbors=max_neighbors
    )

    output_path = Path(output)
    graph.write_graphml(str(output_path))
    logging.info(f"Similarity graph saved to {output_path}")


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
            help="Show the version and exit.",
        )
):
    pass

def _parse_output_db(output_db: str) -> tuple[Path, str, str]:
    """Parse output_db into directory, index name, and resolved path string."""
    output_db_path = Path(output_db)
    db_dir = output_db_path.parent
    index_name = output_db_path.name

    if not index_name:
        index_name = "embeddings"
    if db_dir == Path('.'):
        db_dir = Path.cwd()

    return db_dir, index_name, str(db_dir / index_name)


def _load_embeddings_from_dir(embedding_dir: str, file_extension: Optional[str] = None) -> tuple[list, list]:
    """Load embedding IDs and tensors from a directory of .csv/.pt files."""
    embedding_path = Path(embedding_dir)
    if not embedding_path.exists():
        raise ValueError(f"Embedding directory does not exist: {embedding_dir}")

    if file_extension is not None:
        extensions = [file_extension]
    else:
        extensions = ['.csv', '.pt']

    embedding_files = []
    for ext in extensions:
        embedding_files.extend(sorted(embedding_path.glob(f"*{ext}")))

    if not embedding_files:
        raise ValueError(f"No embedding files found with extensions {extensions} in {embedding_dir}")

    chain_ids = []
    embeddings = []
    for emb_file in embedding_files:
        chain_id = emb_file.stem
        if emb_file.suffix == '.pt':
            embedding = torch.load(emb_file, map_location='cpu', weights_only=True)
        else:
            df = pd.read_csv(emb_file, header=None)
            embedding = torch.tensor(df.values, dtype=torch.float32).squeeze()
        chain_ids.append(chain_id)
        embeddings.append(embedding)

    return chain_ids, embeddings


def _compute_fasta_embeddings(
        fasta_file, tmp_dir, min_res_n, compute_residue_embedding,
        accelerator, devices, strategy,
        batch_size, num_workers, num_nodes,
        batch_size_aggregator, num_workers_aggregator, num_nodes_aggregator
) -> tuple[list, list]:
    """Compute chain embeddings from a FASTA file and return (chain_ids, embeddings)."""
    from foldmatch.inference.sequence_inference import predict as sequence_predict
    from foldmatch.inference.chain_inference import predict as chain_predict
    from foldmatch.cli.sequence import scan_fasta_sequences
    from foldmatch.types.api_types import SrcLocation, SrcTensorFrom, OutFormat

    dev = arg_devices(devices)

    if compute_residue_embedding:
        logging.info("Computing residue embeddings from FASTA sequences...")
        sequence_predict(
            fasta_file=fasta_file,
            min_res_n=min_res_n,
            batch_size=batch_size,
            num_workers=num_workers,
            num_nodes=num_nodes,
            accelerator=accelerator,
            devices=dev,
            out_format=OutFormat.separated,
            out_path=tmp_dir,
            strategy=strategy,
            write_tensor=True
        )

    logging.info("Computing chain embeddings...")
    src_stream = scan_fasta_sequences(fasta_file, tmp_dir)
    structure_embeddings = chain_predict(
        src_stream=src_stream,
        src_location=SrcLocation.stream,
        src_from=SrcTensorFrom.file,
        batch_size=batch_size_aggregator,
        num_workers=num_workers_aggregator,
        num_nodes=num_nodes_aggregator,
        accelerator=accelerator,
        devices=dev,
        strategy=strategy
    )

    chain_ids = [ch_id for _, chain_ids_batch in structure_embeddings for ch_id in chain_ids_batch]
    embeddings = [embedding for embedding_tensor, _ in structure_embeddings for embedding in torch.split(embedding_tensor, 1, dim=0)]

    logging.info(f"Computed {len(embeddings)} chain embeddings")
    return chain_ids, embeddings


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

    logging.debug(f"Filtering results with similarity score threshold >= {threshold}")
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
    logging.debug(f"Filtered from {total_before} to {total_after} results")
    return filtered_results


if __name__ == "__main__":
    app()
