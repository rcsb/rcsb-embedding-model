import logging
import time
from pathlib import Path
from typing import Optional

import torch

from foldmatch.types.api_types import StructureFormat, Accelerator, Granularity
from foldmatch.search.faiss_database import FaissEmbeddingDatabase
from foldmatch.search.embedding_computer import EmbeddingComputer, _is_rank_zero

logger = logging.getLogger(__name__)


class EmbeddingDatabaseBuilder:
    """Compute embeddings (from structures or FASTA) and persist them to a FAISS database.

    On multi-GPU runs every rank participates in the embedding computation;
    only rank 0 materializes the FAISS database.
    """

    def __init__(self, tmp_dir: str, accelerator: Accelerator = 'auto'):
        """
        Args:
            tmp_dir: Directory under which per-run scratch directories are created.
            accelerator: Device to use for inference.
        """
        self.computer = EmbeddingComputer(tmp_dir=tmp_dir, accelerator=accelerator)

    def build_from_structures(
            self,
            structure_dir: str,
            output_db: str,
            structure_format: StructureFormat = StructureFormat.mmcif,
            min_res: int = 10,
            granularity: Granularity = 'chain',
            file_extension: Optional[str] = None,
            use_gpu_index: bool = False,
            batch_size_res: int = 1,
            num_workers_res: int = 0,
            num_nodes_res: int = 1,
            batch_size_chain: int = 1,
            num_workers_chain: int = 0,
            num_nodes_chain: int = 1,
            devices='auto',
            strategy='auto',
    ):
        """Build a FAISS database from a directory of structure files."""
        logging.info("Building embeddings and FAISS database from structures")
        start_time = time.time()
        chain_ids, embeddings = self.computer.compute_from_structures(
            structure_dir=structure_dir,
            structure_format=structure_format,
            min_res=min_res,
            granularity=granularity,
            file_extension=file_extension,
            batch_size_res=batch_size_res,
            num_workers_res=num_workers_res,
            num_nodes_res=num_nodes_res,
            batch_size_chain=batch_size_chain,
            num_workers_chain=num_workers_chain,
            num_nodes_chain=num_nodes_chain,
            devices=devices,
            strategy=strategy,
        )
        self._create(output_db, chain_ids, embeddings, use_gpu_index, start_time)

    def build_from_fasta(
            self,
            fasta_file: str,
            output_db: str,
            min_res_n: int = 0,
            use_gpu_index: bool = False,
            batch_size_res: int = 1,
            num_workers_res: int = 0,
            num_nodes_res: int = 1,
            batch_size_chain: int = 1,
            num_workers_chain: int = 0,
            num_nodes_chain: int = 1,
            devices='auto',
            strategy='auto',
    ):
        """Build a FAISS database from protein sequences in a FASTA file."""
        logging.info("Building embeddings and FAISS database from FASTA")
        start_time = time.time()
        chain_ids, embeddings = self.computer.compute_from_fasta(
            fasta_file=fasta_file,
            min_res_n=min_res_n,
            batch_size_res=batch_size_res,
            num_workers_res=num_workers_res,
            num_nodes_res=num_nodes_res,
            batch_size_chain=batch_size_chain,
            num_workers_chain=num_workers_chain,
            num_nodes_chain=num_nodes_chain,
            devices=devices,
            strategy=strategy,
        )
        self._create(output_db, chain_ids, embeddings, use_gpu_index, start_time)

    def update_from_structures(
            self,
            structure_dir: str,
            output_db: str,
            structure_format: StructureFormat = StructureFormat.mmcif,
            min_res: int = 10,
            granularity: Granularity = 'chain',
            file_extension: Optional[str] = None,
            use_gpu_index: bool = False,
            batch_size_res: int = 1,
            num_workers_res: int = 0,
            num_nodes_res: int = 1,
            batch_size_chain: int = 1,
            num_workers_chain: int = 0,
            num_nodes_chain: int = 1,
            devices='auto',
            strategy='auto',
    ):
        """Update an existing FAISS database with embeddings from structure files."""
        logging.info("Updating FAISS database with embeddings from structures")
        start_time = time.time()
        chain_ids, embeddings = self.computer.compute_from_structures(
            structure_dir=structure_dir,
            structure_format=structure_format,
            min_res=min_res,
            granularity=granularity,
            file_extension=file_extension,
            batch_size_res=batch_size_res,
            num_workers_res=num_workers_res,
            num_nodes_res=num_nodes_res,
            batch_size_chain=batch_size_chain,
            num_workers_chain=num_workers_chain,
            num_nodes_chain=num_nodes_chain,
            devices=devices,
            strategy=strategy,
        )
        self._update(output_db, chain_ids, embeddings, use_gpu_index, start_time)

    def update_from_fasta(
            self,
            fasta_file: str,
            output_db: str,
            min_res_n: int = 0,
            use_gpu_index: bool = False,
            batch_size_res: int = 1,
            num_workers_res: int = 0,
            num_nodes_res: int = 1,
            batch_size_chain: int = 1,
            num_workers_chain: int = 0,
            num_nodes_chain: int = 1,
            devices='auto',
            strategy='auto',
    ):
        """Update an existing FAISS database with embeddings from FASTA sequences."""
        logging.info("Updating FAISS database with embeddings from FASTA")
        start_time = time.time()
        chain_ids, embeddings = self.computer.compute_from_fasta(
            fasta_file=fasta_file,
            min_res_n=min_res_n,
            batch_size_res=batch_size_res,
            num_workers_res=num_workers_res,
            num_nodes_res=num_nodes_res,
            batch_size_chain=batch_size_chain,
            num_workers_chain=num_workers_chain,
            num_nodes_chain=num_nodes_chain,
            devices=devices,
            strategy=strategy,
        )
        self._update(output_db, chain_ids, embeddings, use_gpu_index, start_time)

    def _create(self, output_db, chain_ids, embeddings, use_gpu_index, compute_start):
        db_dir, index_name, output_db = _parse_output_db(output_db)
        if _is_rank_zero():
            embeddings_time = time.time() - compute_start
            logging.info(f"Creating embeddings completed in {embeddings_time:.2f} seconds")

            start_time = time.time()
            db = FaissEmbeddingDatabase(db_path=str(db_dir), index_name=index_name)
            db.create_database(chain_ids=chain_ids, embeddings=embeddings, use_gpu=use_gpu_index)
            database_time = time.time() - start_time
            logging.info(f"Creating database completed in {database_time:.2f} seconds")

            logging.info("Database build complete!")
            logging.info(f"Database location: {output_db}")
            logging.info(f"Total embeddings: {len(db.chain_ids)}")

        del chain_ids, embeddings
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _update(self, output_db, chain_ids, embeddings, use_gpu_index, compute_start):
        db_dir, index_name, output_db = _parse_output_db(output_db)
        if _is_rank_zero():
            embeddings_time = time.time() - compute_start
            logging.info(f"Creating embeddings completed in {embeddings_time:.2f} seconds")

            start_time = time.time()
            db = FaissEmbeddingDatabase(db_path=str(db_dir), index_name=index_name)
            db.load_database()
            existing_count = len(db.chain_ids)
            logging.info(f"Existing database contains {existing_count} embeddings")
            db.update_embeddings(chain_ids=chain_ids, embeddings=embeddings, use_gpu=use_gpu_index)
            database_time = time.time() - start_time
            logging.info(f"Updating database completed in {database_time:.2f} seconds")

            logging.info("Database update complete!")
            logging.info(f"Database location: {output_db}")
            logging.info(f"Total embeddings: {len(db.chain_ids)}")

        del chain_ids, embeddings
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _parse_output_db(output_db: str) -> tuple[Path, str, str]:
    """Split a database path into (directory, index name, resolved path)."""
    output_db_path = Path(output_db)
    db_dir = output_db_path.parent
    index_name = output_db_path.name or "embeddings"
    if db_dir == Path('.'):
        db_dir = Path.cwd()
    return db_dir, index_name, str(db_dir / index_name)
