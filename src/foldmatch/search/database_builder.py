import logging
import os
import tempfile

import torch
import torch.distributed as dist
import time
from pathlib import Path
from typing import Optional

from foldmatch.inference.esm_inference import predict as esm_predict
from foldmatch.inference.chain_inference import predict as chain_predict
from foldmatch.inference.assembly_inferece import predict as assembly_predict
from foldmatch.types.api_types import StructureFormat, SrcLocation, SrcProteinFrom, Accelerator, Granularity, SrcAssemblyFrom
from foldmatch.search.faiss_database import FaissEmbeddingDatabase

logger = logging.getLogger(__name__)

class EmbeddingDatabaseBuilder:
    """Build a database of structure chain embeddings from a directory of structure files."""

    def __init__(
            self,
            structure_dir: str,
            tmp_dir: str,
            structure_format: StructureFormat = StructureFormat.mmcif,
            min_res: int = 10,
            accelerator: Accelerator = 'auto'
    ):
        """
        Initialize the database builder.

        Args:
            structure_dir: Directory containing structure files
            structure_format: Format of structure files (mmcif or pdb)
            min_res: Minimum residue length for chains
            accelerator: Device to use for computation
        """
        self.structure_dir = Path(structure_dir)
        tmp_dir = tempfile.mkdtemp(prefix="run_", dir=tmp_dir)
        self.tmp_res_dir = Path(tmp_dir) / "res"
        os.makedirs(self.tmp_res_dir, exist_ok=True)
        self.tmp_ch_dir = Path(tmp_dir) / "ch"
        os.makedirs(self.tmp_ch_dir, exist_ok=True)
        self.structure_format = structure_format
        self.min_res_n = min_res
        self.accelerator = accelerator

    def build_embeddings(
            self,
            file_extension: Optional[str] = None,
            granularity: Granularity = 'chain',
            devices='auto',
            strategy='auto',
            batch_size_res=1,
            num_workers_res=0,
            num_nodes_res=1,
            batch_size_chain=1,
            num_workers_chain=0,
            num_nodes_chain=1,
    ):
        """
        Build embeddings from structure files in batches (generator).

        Args:
            file_extension: File extension filter (e.g., '.cif', '.pdb'). If None, uses structure_format default
            granularity: Calculate embeddings for 'chain' or 'assembly' level
            devices: Number of devices to use for inference
            strategy: Lightning strategy to control distribution of inference
            batch_size_res: Number of chains to process residue embeddings per batch
            num_workers_res: Number of subprocesses to use for residue embedding data loading
            num_nodes_res: Number of nodes to use for residue embedding inference
            batch_size_chain: Number of chains to process chain embeddings per batch
            num_workers_chain: Number of subprocesses to use for chain embedding data loading
            num_nodes_chain: Number of nodes to use for chain embedding inference

        Yields:
            Tuple of (chain_ids, embeddings) for each batch where chain_ids are in format "structure_name:chain_id"
        """
        if file_extension is None:
            file_extension = '.cif' if self.structure_format == StructureFormat.mmcif else '.pdb'

        if not self.structure_dir.exists():
            raise ValueError(f"Structure directory does not exist: {self.structure_dir}")

        # Collect all structure files
        logging.info(f"Listing structure files from: {self.structure_dir}")
        structure_files = list(self.structure_dir.glob(f"*{file_extension}"))
        if not structure_files:
            raise ValueError(f"No structure files found with extension {file_extension} in {self.structure_dir}")

        esm_predict(
            src_stream=[
                (str_file.stem, str_file, str_file.stem)
                for str_file in structure_files
            ],
            src_location=SrcLocation.stream,
            src_from=SrcProteinFrom.structure,
            structure_format=self.structure_format,
            min_res_n=self.min_res_n,
            out_path=self.tmp_res_dir,
            accelerator=self.accelerator,
            batch_size=batch_size_res,
            num_workers=num_workers_res,
            num_nodes=num_nodes_res,
            devices=devices,
            strategy=strategy,
            write_tensor=True
        )
        if _is_distributed():
            dist.barrier()

        logging.info(f"Listing residue embedding files from: {self.structure_dir}")
        esm_embedding_files = list(self.tmp_res_dir.glob(f"*pt"))
        if granularity == 'chain':
            chain_predict(
                src_stream=[
                    (esm_file, esm_file.stem)
                    for esm_file in esm_embedding_files
                ],
                src_location=SrcLocation.stream,
                out_path=self.tmp_ch_dir,
                accelerator=self.accelerator,
                batch_size=batch_size_chain,
                num_workers=num_workers_chain,
                num_nodes=num_nodes_chain,
                devices=devices,
                strategy=strategy,
                write_tensor=True
            )
        else:
            assembly_predict(
                src_stream=[
                    (str_file.stem, str_file, str_file.stem)
                    for str_file in structure_files
                ],
                res_embedding_location=str(self.tmp_res_dir),
                src_location=SrcLocation.stream,
                out_path=self.tmp_ch_dir,
                src_from=SrcAssemblyFrom.structure,
                accelerator=self.accelerator,
                num_workers=num_workers_chain,
                num_nodes=num_nodes_chain,
                devices=devices,
                strategy=strategy,
                write_tensor=True
            )
        if _is_distributed():
            dist.barrier()

        tensor_files = [f for f in self.tmp_ch_dir.iterdir() if f.is_file()]
        names = [f.stem for f in tensor_files]
        embeddings = [torch.load(f) for f in tensor_files]

        return names, embeddings

    def build_faiss_database(
            self,
            output_db: str,
            file_extension: Optional[str] = None,
            granularity: Granularity = 'chain',
            use_gpu_index: bool = False,
            batch_size_res=1,
            num_workers_res=0,
            num_nodes_res=1,
            batch_size_chain=1,
            num_workers_chain=0,
            num_nodes_chain=1,
            devices='auto',
            strategy='auto'
    ):
        """
        Build a FAISS database from structure files in batches.

        Args:
            output_db: Path to save the FAISS database (directory + prefix)
            file_extension: File extension filter (e.g., '.cif', '.pdb'). If None, uses structure_format default
            granularity: Calculate embeddings for 'chain' or 'assembly' level
            use_gpu_index: Whether to use GPU for FAISS indexing
            batch_size_res: Number of chains to process residue embeddings per batch
            num_workers_res: Number of subprocesses to use for residue embedding data loading
            num_nodes_res: Number of nodes to use for residue embedding inference
            batch_size_chain: Number of chains to process chain embeddings per batch
            num_workers_chain: Number of subprocesses to use for chain embedding data loading
            num_nodes_chain: Number of nodes to use for chain embedding inference
            devices: Number of devices to use for inference
            strategy: Lightning strategy to control distribution of inference
        """
        # Parse output_db into directory and prefix
        output_db_path = Path(output_db)
        db_dir = output_db_path.parent
        index_name = output_db_path.name

        # Ensure we have a valid directory and prefix
        if not index_name:
            index_name = "embeddings"
        if db_dir == Path('.'):
            db_dir = Path.cwd()

        logging.info("Building embeddings and FAISS database")

        db = FaissEmbeddingDatabase(db_path=str(db_dir), index_name=index_name)

        start_time = time.time()
        chain_ids, embeddings = self.build_embeddings(
                file_extension=file_extension,
                granularity=granularity,
                devices=devices,
                strategy=strategy,
                batch_size_res=batch_size_res,
                num_workers_res=num_workers_res,
                num_nodes_res=num_nodes_res,
                batch_size_chain=batch_size_chain,
                num_workers_chain=num_workers_chain,
                num_nodes_chain=num_nodes_chain,
        )

        if _is_rank_zero():
            embeddings_time = time.time() - start_time
            logging.info(f"Creating embeddings completed in {embeddings_time:.2f} seconds")

            start_time = time.time()
            db.create_database(chain_ids=chain_ids, embeddings=embeddings, use_gpu=use_gpu_index)
            database_time = time.time() - start_time
            logging.info(f"Creating database completed in {database_time:.2f} seconds")

            logging.info("Batch database build complete!")
            logging.info(f"Database location: {output_db}")
            logging.info(f"Total embeddings: {len(db.chain_ids)}")

        del chain_ids, embeddings
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def update_faiss_database(
            self,
            output_db: str,
            file_extension: Optional[str] = None,
            granularity: Granularity = 'chain',
            use_gpu_index: bool = False,
            batch_size_res=1,
            num_workers_res=0,
            num_nodes_res=1,
            batch_size_chain=1,
            num_workers_chain=0,
            num_nodes_chain=1,
            devices='auto',
            strategy='auto'
    ):
        """
        Update an existing FAISS database with new or replacement embeddings.

        New embeddings are added to the database. If any of the new IDs match
        existing entries, their vectors are replaced.

        Args:
            output_db: Path to the existing FAISS database (directory + prefix)
            file_extension: File extension filter (e.g., '.cif', '.pdb'). If None, uses structure_format default
            granularity: Calculate embeddings for 'chain' or 'assembly' level
            use_gpu_index: Whether to use GPU for FAISS indexing
            batch_size_res: Number of chains to process residue embeddings per batch
            num_workers_res: Number of subprocesses to use for residue embedding data loading
            num_nodes_res: Number of nodes to use for residue embedding inference
            batch_size_chain: Number of chains to process chain embeddings per batch
            num_workers_chain: Number of subprocesses to use for chain embedding data loading
            num_nodes_chain: Number of nodes to use for chain embedding inference
            devices: Number of devices to use for inference
            strategy: Lightning strategy to control distribution of inference
        """
        # Parse output_db into directory and prefix
        output_db_path = Path(output_db)
        db_dir = output_db_path.parent
        index_name = output_db_path.name

        if not index_name:
            index_name = "embeddings"
        if db_dir == Path('.'):
            db_dir = Path.cwd()

        logging.info("Loading existing FAISS database for update")

        db = FaissEmbeddingDatabase(db_path=str(db_dir), index_name=index_name)
        db.load_database()

        existing_count = len(db.chain_ids)
        logging.info(f"Existing database contains {existing_count} embeddings")

        logging.info("Building new embeddings")

        start_time = time.time()
        chain_ids, embeddings = self.build_embeddings(
            file_extension=file_extension,
            granularity=granularity,
            devices=devices,
            strategy=strategy,
            batch_size_res=batch_size_res,
            num_workers_res=num_workers_res,
            num_nodes_res=num_nodes_res,
            batch_size_chain=batch_size_chain,
            num_workers_chain=num_workers_chain,
            num_nodes_chain=num_nodes_chain,
        )
        if _is_rank_zero():
            embeddings_time = time.time() - start_time
            logging.info(f"Creating embeddings completed in {embeddings_time:.2f} seconds")

            start_time = time.time()
            db.update_embeddings(chain_ids=chain_ids, embeddings=embeddings, use_gpu=use_gpu_index)
            database_time = time.time() - start_time
            logging.info(f"Updating database completed in {database_time:.2f} seconds")

            logging.info("Database update complete!")
            logging.info(f"Database location: {output_db}")
            logging.info(f"Total embeddings: {len(db.chain_ids)}")

        del chain_ids, embeddings
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def _is_distributed():
    """Check if the current process is running in distributed mode."""
    return dist.is_available() and dist.is_initialized()

def _is_rank_zero():
    """Check if the current process is rank zero in distributed training."""
    return not _is_distributed() or dist.get_rank() == 0
