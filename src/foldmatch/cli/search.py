import os
import logging
import typer
from pathlib import Path
from typing import Annotated, Optional, Tuple

from foldmatch import __version__
from foldmatch.cli.args_utils import arg_devices, set_log_level
from foldmatch.types.api_types import (
    StructureFormat,
    Accelerator,
    Strategy,
    Granularity,
    IndexConfig,
    IndexType,
    LogLevel,
)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = typer.Typer(
    add_completion=False,
    pretty_exceptions_enable=False,
    help=f"3D structure search using embeddings and FAISS. Version: {__version__}."
)

build_db_app = typer.Typer(
    add_completion=False,
    help="Build an embedding database."
)
app.add_typer(build_db_app, name="build")

query_db_app = typer.Typer(
    add_completion=False,
    help="Query an existing embedding database."
)
app.add_typer(query_db_app, name="query")


@build_db_app.command(
    name="structures",
    help="Build an embedding database from a directory of structure files."
)
def build_database_from_structures(
        structure_folder: Annotated[Path, typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help='Folder containing structure files.'
        )],
        output_db: Annotated[str, typer.Option(
            help='Path to save the FAISS database.'
        )],
        tmp_embedding_folder: Annotated[Path, typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help='Folder for intermediate embeddings.'
        )],
        structure_format: Annotated[StructureFormat, typer.Option(
            help='Structure file format (mmcif, binarycif, or pdb).'
        )] = StructureFormat.mmcif,
        granularity: Annotated[Granularity, typer.Option(
            help='Calculate embeddings for "chain" or "assembly" level.'
        )] = Granularity.chain,
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
        )] = Accelerator.auto,
        devices: Annotated[Tuple[str], typer.Option(
            help='The devices to use. Can be set to a positive number or "auto". Repeat this argument to indicate multiple indices of devices. "auto" for automatic selection based on the chosen accelerator.'
        )] = ('auto',),
        num_nodes: Annotated[int, typer.Option(
            help='Number of nodes to use for inference.'
        )] = 1,
        strategy: Annotated[Strategy, typer.Option(
            help='Lightning strategy to control distribution of inference.'
        )] = Strategy.auto,
        batch_size: Annotated[int, typer.Option(
            help='Number of samples processed together in one iteration.'
        )] = 1,
        num_workers: Annotated[int, typer.Option(
            help='Number of subprocesses to use for data loading.'
        )] = 0,
        index_type: Annotated[IndexType, typer.Option(
            help='FAISS index variant: auto (legacy heuristic), flat, hnsw, or ivf_pq (on-disk).'
        )] = IndexType.auto,
        ivf_nlist: Annotated[int, typer.Option(
            help='IVF cell count for ivf_pq index.'
        )] = 16384,
        ivf_nprobe: Annotated[Optional[int], typer.Option(
            help='Cells probed per query for ivf_pq (defaults to nlist // 64).'
        )] = None,
        load_batch_size: Annotated[int, typer.Option(
            help='Number of vectors per batch streamed into the FAISS index. Larger '
                 'batches (e.g. 65536+) better saturate many-core nodes during '
                 'IVF-PQ build; smaller batches use less memory.'
        )] = 32768,
        log_level: Annotated[LogLevel, typer.Option(
            help='Number of nodes to use for inference of embeddings.'
        )] = LogLevel.info
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

    if use_gpu_index:
        logging.info("GPU acceleration for FAISS index: enabled")

    from foldmatch.search.embedding_computer import EmbeddingComputer
    embedding_computer = EmbeddingComputer(
        embedding_folder=tmp_embedding_folder,
    )
    embedding_computer.compute_from_structures(
        structure_folder=structure_folder,
        structure_format=structure_format,
        min_res=min_res,
        granularity=granularity,
        file_extension=file_extension,
        batch_size=batch_size,
        num_workers=num_workers,
        num_nodes=num_nodes,
        accelerator=accelerator,
        devices=arg_devices(devices),
        strategy=strategy,
    )

    if not _is_rank_zero():
        return

    from foldmatch.search.embedding_database import EmbeddingDatabase
    embedding_db = EmbeddingDatabase(
        db_path=output_db
    )
    embedding_db.create_db(
        embedding_batches=embedding_computer.get_embedding_batches(load_batch_size=load_batch_size),
        use_gpu_index=use_gpu_index,
        index_type=index_type,
        index_config=IndexConfig(nlist=ivf_nlist, nprobe=ivf_nprobe),
    )


@build_db_app.command(
    name="embeddings",
    help="Build an embedding database from a directory of pre-computed embedding files (.pt, .csv, or .parquet)."
)
def build_database_from_embeddings(
        embedding_folder: Annotated[Path, typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help='Folder containing pre-computed embedding files (.pt, .csv, or .parquet).'
        )],
        output_db: Annotated[str, typer.Option(
            help='Path to save the FAISS database.'
        )],
        file_extension: Annotated[Optional[str], typer.Option(
            help='Restrict loading to a single extension (.pt, .csv, or .parquet). If unset, all three are collected.'
        )] = None,
        use_gpu_index: Annotated[bool, typer.Option(
            help='Use GPU for FAISS index (requires faiss-gpu).'
        )] = False,
        index_type: Annotated[IndexType, typer.Option(
            help='FAISS index variant: auto (legacy heuristic), flat, hnsw, or ivf_pq (on-disk).'
        )] = IndexType.auto,
        ivf_nlist: Annotated[int, typer.Option(
            help='IVF cell count for ivf_pq index.'
        )] = 16384,
        ivf_nprobe: Annotated[Optional[int], typer.Option(
            help='Cells probed per query for ivf_pq (defaults to nlist // 64).'
        )] = None,
        load_batch_size: Annotated[int, typer.Option(
            help='Number of vectors per batch streamed into the FAISS index. Larger '
                 'batches (e.g. 65536+) better saturate many-core nodes during '
                 'IVF-PQ build; smaller batches use less memory.'
        )] = 32768,
        log_level: Annotated[LogLevel, typer.Option(
            help='Logging level.'
        )] = LogLevel.info
):
    """Build an embedding database from pre-computed embedding files."""

    set_log_level(log_level)

    from foldmatch.search.embedding_database import EmbeddingDatabase
    embedding_db = EmbeddingDatabase(
        db_path=output_db
    )
    from foldmatch.utils.data import stream_embeddings
    embedding_db.create_db(
        embedding_batches=stream_embeddings(
            path=embedding_folder,
            file_extension=file_extension,
            batch_size=load_batch_size
        ),
        use_gpu_index=use_gpu_index,
        index_type=index_type,
        index_config=IndexConfig(nlist=ivf_nlist, nprobe=ivf_nprobe),
    )


@build_db_app.command(
    name="sequences",
    help="Build an embedding database from protein sequences in a FASTA file."
)
def build_database_from_fasta(
        fasta_file: Annotated[Path, typer.Option(
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
            help='FASTA file containing protein sequences.'
        )],
        output_db: Annotated[str, typer.Option(
            help='Path to save the FAISS database.'
        )],
        tmp_embedding_folder: Annotated[Path, typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help='Directory for intermediate embeddings.'
        )],
        min_res_n: Annotated[int, typer.Option(
            help='Consider only sequences with at least <min_res_n> residues.'
        )] = 0,
        use_gpu_index: Annotated[bool, typer.Option(
            help='Use GPU for FAISS index (requires faiss-gpu).'
        )] = False,
        accelerator: Annotated[Accelerator, typer.Option(
            help='Device used for inference.'
        )] = Accelerator.auto,
        num_nodes: Annotated[int, typer.Option(
            help='Number of nodes to use for inference.'
        )] = 1,
        devices: Annotated[Tuple[str], typer.Option(
            help='The devices to use. Can be set to a positive number or "auto".'
        )] = ('auto',),
        strategy: Annotated[Strategy, typer.Option(
            help='Lightning strategy to control distribution of inference.'
        )] = Strategy.auto,
        batch_size: Annotated[int, typer.Option(
            help='Number of samples processed together for residue embedding inference.'
        )] = 1,
        num_workers: Annotated[int, typer.Option(
            help='Number of subprocesses to use for data loading.'
        )] = 0,
        index_type: Annotated[IndexType, typer.Option(
            help='FAISS index variant: auto (legacy heuristic), flat, hnsw, or ivf_pq (on-disk).'
        )] = IndexType.auto,
        ivf_nlist: Annotated[int, typer.Option(
            help='IVF cell count for ivf_pq index.'
        )] = 16384,
        ivf_nprobe: Annotated[Optional[int], typer.Option(
            help='Cells probed per query for ivf_pq (defaults to nlist // 64).'
        )] = None,
        load_batch_size: Annotated[int, typer.Option(
            help='Number of vectors per batch streamed into the FAISS index. Larger '
                 'batches (e.g. 65536+) better saturate many-core nodes during '
                 'IVF-PQ build; smaller batches use less memory.'
        )] = 32768,
        log_level: Annotated[LogLevel, typer.Option(
            help='Logging level.'
        )] = LogLevel.info
):
    """Build an embedding database from protein sequences in a FASTA file."""

    set_log_level(log_level)

    from foldmatch.search.embedding_computer import EmbeddingComputer
    embedding_computer = EmbeddingComputer(
        embedding_folder=tmp_embedding_folder,
    )
    """embedding_computer.compute_from_fasta(
        fasta_file=fasta_file,
        min_res_n=min_res_n,
        batch_size=batch_size,
        num_workers=num_workers,
        num_nodes=num_nodes,
        accelerator=accelerator,
        devices=arg_devices(devices),
        strategy=strategy,
    )"""

    if not _is_rank_zero():
        return

    from foldmatch.search.embedding_database import EmbeddingDatabase
    embedding_db = EmbeddingDatabase(
        db_path=output_db
    )
    embedding_db.create_db(
        embedding_batches=embedding_computer.get_embedding_batches(load_batch_size=load_batch_size),
        use_gpu_index=use_gpu_index,
        index_type=index_type,
        index_config=IndexConfig(nlist=ivf_nlist, nprobe=ivf_nprobe),
    )

    # Persist a sidecar id->sequence store so this database can be the
    # subject (or query) of a Stage-2 pairwise-alignment search. Built
    # directly from the FASTA — the DB ids ARE the FASTA headers — using the
    # same min_res_n filter so store and index stay in lock-step.
    from foldmatch.search.sequence_store import SequenceStore
    SequenceStore(embedding_db.db_folder, embedding_db.index_name).create(
        fasta_file=fasta_file,
        min_res_n=min_res_n,
    )


@query_db_app.command(
    name="structure",
    help="Search the database using a structure file."
)
def query_database_from_structure(
        db_path: Annotated[str, typer.Option(
            help='Path to the FAISS database.'
        )],
        query_structure: Annotated[Path, typer.Option(
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
            help='Path to query structure file.'
        )],
        structure_format: Annotated[StructureFormat, typer.Option(
            help='Structure file format (mmcif, binarycif or pdb)'
        )] = StructureFormat.mmcif,
        granularity: Annotated[Granularity, typer.Option(
            help='Query database for "chain" or "assembly" embeddings.'
        )] = Granularity.chain,
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
        accelerator: Annotated[Accelerator, typer.Option(
            help='Device used for inference.'
        )] = Accelerator.auto,
        use_gpu_index: Annotated[bool, typer.Option(
            help='Use GPU for FAISS search (requires faiss-gpu).'
        )] = False,
        nprobe: Annotated[Optional[int], typer.Option(
            help='Override the search-time nprobe (cells probed per query) for '
                 'ivf_pq indexes. Higher = better recall, slower search. '
                 'Defaults to the value baked in at build time. '
                 'Ignored for flat/hnsw indexes.'
        )] = None,
        log_level: Annotated[LogLevel, typer.Option(
            help='Number of nodes to use for inference of embeddings.'
        )] = LogLevel.info
):
    """Search database for similar structures."""

    set_log_level(log_level)

    from foldmatch.search.embedding_computer import compute_from_structure_file
    embedding_batches = compute_from_structure_file(
        query_structure=query_structure,
        structure_format=structure_format,
        granularity=granularity,
        chain_id=chain_id,
        assembly_id=assembly_id,
        min_res=min_res,
        max_res=max_res,
        accelerator=accelerator
    )

    logging.info("Loading database...")
    from foldmatch.search.embedding_database import EmbeddingDatabase
    embedding_db = EmbeddingDatabase(
        db_path=db_path,
        use_gpu_for_search=use_gpu_index
    )
    embedding_db.load(nprobe=nprobe)
    results = embedding_db.search_by_embeddings(
        embedding_batches=embedding_batches,
        top_k=top_k
    )

    # Filter by threshold if specified
    results = _filter_results_by_threshold(results, threshold)

    # Either print to stdout OR export to CSV — never both. Printing thousands
    # of result rows to stdout can swamp SLURM's I/O pipeline (Abandoning IO
    # warnings) and is redundant when the CSV is the real artifact.
    if output_csv:
        embedding_db.export_results(results, output_csv)
    else:
        embedding_db.print_results(results)


@query_db_app.command(
    name="embedding",
    help="Search the database using a pre-computed embedding file (.pt, .csv, or .parquet)."
)
def query_database_from_embedding(
        db_path: Annotated[str, typer.Option(
            help='Path to the FAISS database.'
        )],
        embedding_folder: Annotated[Path, typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help='Directory containing pre-computed embedding files (.pt, .csv, or .parquet).'
        )],
        file_extension: Annotated[Optional[str], typer.Option(
            help='Restrict loading to a single extension (.pt, .csv, or .parquet). If unset, all three are collected.'
        )] = None,
        top_k: Annotated[int, typer.Option(
            help='Number of top results to return.'
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
        nprobe: Annotated[Optional[int], typer.Option(
            help='Override the search-time nprobe (cells probed per query) for '
                 'ivf_pq indexes. Higher = better recall, slower search. '
                 'Defaults to the value baked in at build time. '
                 'Ignored for flat/hnsw indexes.'
        )] = None,
        log_level: Annotated[LogLevel, typer.Option(
            help='Logging level.'
        )] = LogLevel.info
):
    """Search database using a pre-computed embedding file."""

    set_log_level(log_level)

    logging.info("Loading database...")
    from foldmatch.search.embedding_database import EmbeddingDatabase
    embedding_db = EmbeddingDatabase(
        db_path=db_path,
        use_gpu_for_search=use_gpu_index
    )
    embedding_db.load(nprobe=nprobe)
    from foldmatch.utils.data import stream_embeddings
    results = embedding_db.search_by_embeddings(
        embedding_batches=stream_embeddings(
            path=embedding_folder,
            file_extension=file_extension
        ),
        top_k=top_k
    )

    if not results:
        raise ValueError(f"No embeddings found in {embedding_folder}")

    logging.info(f"Found {len(results)} results from {embedding_folder}")
    results = _filter_results_by_threshold(results, threshold)

    # See query_database_from_structure for the rationale: print xor CSV.
    if output_csv:
        embedding_db.export_results(results, output_csv)
    else:
        embedding_db.print_results(results)


@query_db_app.command(
    name="sequences",
    help="Search the database using protein sequences from a FASTA file. Each sequence is used as a separate query."
)
def query_database_from_fasta(
        db_path: Annotated[str, typer.Option(
            help='Path to the FAISS database.'
        )],
        fasta_file: Annotated[Path, typer.Option(
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
            help='FASTA file containing protein sequences. Each sequence is used as a separate query.'
        )],
        tmp_embedding_folder: Annotated[Path, typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help='Directory for intermediate embeddings.'
        )],
        min_res_n: Annotated[int, typer.Option(
            help='Consider only sequences with at least <min_res_n> residues.'
        )] = 0,
        top_k: Annotated[int, typer.Option(
            help='Number of top results to return per query.'
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
        accelerator: Annotated[Accelerator, typer.Option(
            help='Device used for inference.'
        )] = Accelerator.auto,
        num_nodes: Annotated[int, typer.Option(
            help='Number of nodes to use for inference.'
        )] = 1,
        devices: Annotated[Tuple[str], typer.Option(
            help='The devices to use. Can be set to a positive number or "auto".'
        )] = ('auto',),
        strategy: Annotated[Strategy, typer.Option(
            help='Lightning strategy to control distribution of inference.'
        )] = Strategy.auto,
        batch_size: Annotated[int, typer.Option(
            help='Number of samples processed together for residue embedding inference.'
        )] = 1,
        num_workers: Annotated[int, typer.Option(
            help='Number of subprocesses to use for data loading.'
        )] = 0,
        nprobe: Annotated[Optional[int], typer.Option(
            help='Override the search-time nprobe (cells probed per query) for '
                 'ivf_pq indexes. Higher = better recall, slower search. '
                 'Defaults to the value baked in at build time. '
                 'Ignored for flat/hnsw indexes.'
        )] = None,
        seq_identity: Annotated[bool, typer.Option(
            help='Stage 2: after the embedding prefilter, pairwise-align each '
                 'query against its candidate subjects and report exact sequence '
                 'identity (requires a subject database built from FASTA). Also '
                 'reports Pvalue_approx/Evalue_approx — an APPROXIMATE, relative-'
                 'only significance signal (sampled lambda/K), not BLAST-comparable.'
        )] = False,
        min_seq_identity: Annotated[float, typer.Option(
            help='When --seq-identity is set, drop hits whose alignment identity '
                 'is below this fraction (0-1).'
        )] = 0.3,
        min_coverage: Annotated[float, typer.Option(
            help='When --seq-identity is set, drop hits whose query AND subject '
                 'coverage are below this fraction (0-1).'
        )] = 0.0,
        gap_open: Annotated[int, typer.Option(
            help='Gap-open penalty (positive) for Stage-2 alignment.'
        )] = 11,
        gap_extend: Annotated[int, typer.Option(
            help='Gap-extend penalty (positive) for Stage-2 alignment.'
        )] = 1,
        align_workers: Annotated[Optional[int], typer.Option(
            help='Process-pool size for Stage-2 alignment. Defaults to all CPUs '
                 'on the node (widening any scheduler --cpu-bind pinning); set '
                 'to 0 or 1 to run serially.'
        )] = None,
        log_level: Annotated[LogLevel, typer.Option(
            help='Logging level.'
        )] = LogLevel.info
):
    """Search database using protein sequences from a FASTA file."""

    set_log_level(log_level)

    from foldmatch.search.embedding_computer import EmbeddingComputer
    embedding_computer = EmbeddingComputer(
        embedding_folder=tmp_embedding_folder,
    )
    embedding_computer.compute_from_fasta(
        fasta_file=fasta_file,
        min_res_n=min_res_n,
        batch_size=batch_size,
        num_workers=num_workers,
        num_nodes=num_nodes,
        accelerator=accelerator,
        devices=arg_devices(devices),
        strategy=strategy,
    )

    if not _is_rank_zero():
        return

    logging.info("Loading database...")
    from foldmatch.search.embedding_database import EmbeddingDatabase
    embedding_db = EmbeddingDatabase(
        db_path=db_path,
        use_gpu_for_search=use_gpu_index,
    )
    embedding_db.load(nprobe=nprobe)
    results = embedding_db.search_by_embeddings(
        embedding_batches=embedding_computer.get_embedding_batches(),
        top_k=top_k
    )
    logging.info(f"Searched {len(results)} sequence(s)")
    results = _filter_results_by_threshold(results, threshold)

    if seq_identity:
        from foldmatch.utils.fasta import iter_fasta
        query_sequences = dict(iter_fasta(fasta_file))
        _stage2_align_and_report(
            embedding_db=embedding_db,
            prefilter_results=results,
            query_sequences=query_sequences,
            min_seq_identity=min_seq_identity,
            min_coverage=min_coverage,
            gap_open=gap_open,
            gap_extend=gap_extend,
            num_workers=align_workers,
            output_csv=output_csv,
        )
    elif output_csv:
        embedding_db.export_results(results, output_csv)
    else:
        embedding_db.print_results(results)


@query_db_app.command(
    name="db",
    help="Compare entries from a query database with a subject database."
)
def query_database_from_database(
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
        nprobe: Annotated[Optional[int], typer.Option(
            help='Override the search-time nprobe (cells probed per query) for '
                 'ivf_pq indexes. Higher = better recall, slower search. '
                 'Defaults to the value baked in at build time. '
                 'Ignored for flat/hnsw indexes.'
        )] = None,
        seq_identity: Annotated[bool, typer.Option(
            help='Stage 2: after the embedding prefilter, pairwise-align each '
                 'query against its candidate subjects and report exact sequence '
                 'identity (requires BOTH databases built from FASTA). Also reports '
                 'Pvalue_approx/Evalue_approx — an APPROXIMATE, relative-only '
                 'significance signal (sampled lambda/K), not BLAST-comparable.'
        )] = False,
        min_seq_identity: Annotated[float, typer.Option(
            help='When --seq-identity is set, drop hits whose alignment identity '
                 'is below this fraction (0-1).'
        )] = 0.3,
        min_coverage: Annotated[float, typer.Option(
            help='When --seq-identity is set, drop hits whose query AND subject '
                 'coverage are below this fraction (0-1).'
        )] = 0.0,
        # Note: Pvalue_approx/Evalue_approx columns are an approximate,
        # relative-only significance signal (sampled lambda/K) — not BLAST-comparable.
        gap_open: Annotated[int, typer.Option(
            help='Gap-open penalty (positive) for Stage-2 alignment.'
        )] = 11,
        gap_extend: Annotated[int, typer.Option(
            help='Gap-extend penalty (positive) for Stage-2 alignment.'
        )] = 1,
        align_workers: Annotated[Optional[int], typer.Option(
            help='Process-pool size for Stage-2 alignment. Defaults to all CPUs '
                 'on the node (widening any scheduler --cpu-bind pinning); set '
                 'to 0 or 1 to run serially.'
        )] = None,
        log_level: Annotated[LogLevel, typer.Option(
            help='Number of nodes to use for inference of embeddings.'
        )] = LogLevel.info
):
    """Search subject database using all entries from query database."""

    set_log_level(log_level)

    if use_gpu_index:
        logging.info("GPU acceleration for FAISS search: enabled")

    # Load subject database
    logging.info("\nLoading subject database...")
    from foldmatch.search.embedding_database import EmbeddingDatabase
    embedding_db = EmbeddingDatabase(
        db_path=subject_db_path,
        use_gpu_for_search=use_gpu_index
    )
    embedding_db.load(nprobe=nprobe)

    # Display database statistics
    subject_stats = embedding_db.get_db_statistics()
    logging.info(f"Subject database contains {subject_stats['total_embeddings']} chains")

    # Perform database-to-database search
    logging.info("Performing search...")
    results = embedding_db.search_by_database(
        query_db_path=query_db_path,
        top_k=top_k
    )

    # Filter by threshold if specified
    results = _filter_results_by_threshold(results, threshold)

    if seq_identity:
        # Query sequences come from the query database's own sequence store.
        from foldmatch.search.embedding_database import _parse_db_path
        from foldmatch.search.sequence_store import SequenceStore
        q_folder, q_name, _ = _parse_db_path(query_db_path)
        query_store = SequenceStore(q_folder, q_name)
        if not query_store.exists():
            raise ValueError(
                f"Query database has no sequence store ({query_store.path}); "
                f"Stage-2 sequence identity requires a query database built from "
                f"FASTA sequences (fm-search build sequences)."
            )
        query_sequences = query_store.fetch(results.keys())
        _stage2_align_and_report(
            embedding_db=embedding_db,
            prefilter_results=results,
            query_sequences=query_sequences,
            min_seq_identity=min_seq_identity,
            min_coverage=min_coverage,
            gap_open=gap_open,
            gap_extend=gap_extend,
            num_workers=align_workers,
            output_csv=output_csv,
        )
        return
    elif output_csv:
        embedding_db.export_results(results, output_csv)
    else:
        embedding_db.print_results(results)


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
        )] = LogLevel.info
):
    """Display database statistics."""

    set_log_level(log_level)

    from foldmatch.search.embedding_database import EmbeddingDatabase
    embedding_db = EmbeddingDatabase(
        db_path=db_path
    )
    embedding_db.load()
    stats = embedding_db.get_db_statistics()

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
        )] = LogLevel.info
):
    """Cluster database embeddings using Leiden algorithm."""

    set_log_level(log_level)

    if use_gpu_index:
        logging.info("GPU acceleration for FAISS operations: enabled")

    # Initialize clusterer
    logging.info("Initializing clusterer...")
    from foldmatch.search.clustering import EmbeddingClusterer
    clusterer = EmbeddingClusterer(
        db_path=db_path,
        use_gpu=use_gpu_index
    )

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
        )] = LogLevel.info
):
    """Build and export a similarity graph from database embeddings."""

    set_log_level(log_level)

    if use_gpu_index:
        logging.info("GPU acceleration for FAISS operations: enabled")

    logging.info("Initializing clusterer...")
    from foldmatch.search.clustering import EmbeddingClusterer
    clusterer = EmbeddingClusterer(
        db_path=db_path,
        use_gpu=use_gpu_index
    )

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
def _app_callback(
        version: bool = typer.Option(
            None,
            "--version",
            callback=version_callback,
            is_eager=True,
            help="Show the version and exit.",
        )
):
    pass

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

def _stage2_align_and_report(
        embedding_db,
        prefilter_results,
        query_sequences,
        min_seq_identity: float,
        min_coverage: float,
        gap_open: int,
        gap_extend: int,
        num_workers: Optional[int],
        output_csv: Optional[str],
):
    """Run Stage-2 pairwise alignment on prefilter candidates and report.

    The subject sequences are recovered from the subject database's sidecar
    sequence store; raises a clear error if that store is missing (e.g. a
    database built from structures rather than FASTA).
    """
    from foldmatch.search.sequence_store import SequenceStore
    from foldmatch.search import alignment

    subject_store = SequenceStore(embedding_db.db_folder, embedding_db.index_name)
    if not subject_store.exists():
        raise ValueError(
            f"Subject database has no sequence store ({subject_store.path}); "
            f"Stage-2 sequence identity requires a database built from FASTA "
            f"sequences (fm-search build sequences)."
        )

    logging.info("Computing pairwise sequence alignments (Stage 2)...")
    aligned = alignment.align_candidates(
        query_sequences=query_sequences,
        prefilter_results=prefilter_results,
        fetch_subject_sequences=subject_store.fetch,
        min_seq_identity=min_seq_identity,
        min_coverage=min_coverage,
        gap_open=gap_open,
        gap_extend=gap_extend,
        num_workers=num_workers,
        subject_db_size=subject_store.total_residues(),
    )

    # See query_database_from_structure for the rationale: print xor CSV.
    if output_csv:
        alignment.export_aligned_results(aligned, output_csv)
    else:
        alignment.print_aligned_results(aligned)


def _is_rank_zero():
    """Check if the current process is running in distributed mode."""
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        return True
    return False


def main():
    """Entry point that renders expected errors as clean messages (no traceback)."""
    try:
        app()
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
