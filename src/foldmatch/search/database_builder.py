import logging
import time
from pathlib import Path
from typing import Optional

import torch

from foldmatch.types.api_types import (
    StructureFormat,
    Accelerator,
    Granularity,
    IndexConfig,
    IndexType,
)
from foldmatch.search.faiss_database import FaissEmbeddingDatabase
from foldmatch.search.embedding_computer import EmbeddingComputer, is_rank_zero

logger = logging.getLogger(__name__)


class EmbeddingDatabaseBuilder:
    """Compute embeddings (from structures or FASTA) and persist them to a FAISS database.

    On multi-GPU runs every rank participates in the embedding computation;
    only rank 0 materializes the FAISS database.
    """

    def __init__(self, tmp_dir: Optional[str] = None, accelerator: Accelerator = 'auto'):
        """
        Args:
            tmp_dir: Directory under which per-run scratch directories are created.
                Required for the structure / FASTA build paths (which run inference);
                optional for :meth:`build_from_embeddings` which only reads disk.
            accelerator: Device to use for inference.
        """
        self.tmp_dir = tmp_dir
        self.accelerator = accelerator
        self._computer: Optional[EmbeddingComputer] = None

    @property
    def computer(self) -> EmbeddingComputer:
        """Lazy ``EmbeddingComputer`` — instantiated on first inference-based build."""
        if self._computer is None:
            if self.tmp_dir is None:
                raise ValueError(
                    "tmp_dir is required for inference-based builds "
                    "(build_from_structures, build_from_fasta)"
                )
            self._computer = EmbeddingComputer(
                embedding_folder=self.tmp_dir, accelerator=self.accelerator
            )
        return self._computer

    def build_from_structures(
            self,
            structure_dir: str,
            output_db: str,
            structure_format: StructureFormat = StructureFormat.mmcif,
            min_res: int = 10,
            granularity: Granularity = 'chain',
            file_extension: Optional[str] = None,
            use_gpu_index: bool = False,
            batch_size: int = 1,
            num_workers: int = 0,
            num_nodes: int = 1,
            devices='auto',
            strategy='auto',
            index_type: IndexType = IndexType.auto,
            index_config: Optional[IndexConfig] = None,
    ):
        """Build a FAISS database from a directory of structure files."""
        logging.info("Building embeddings and FAISS database from structures")
        start_time = time.time()
        batches = self.computer.compute_from_structures(
            structure_folder=structure_dir,
            structure_format=structure_format,
            min_res=min_res,
            granularity=granularity,
            file_extension=file_extension,
            batch_size=batch_size,
            num_workers=num_workers,
            num_nodes=num_nodes,
            devices=devices,
            strategy=strategy,
        )
        self._create(output_db, batches, use_gpu_index, index_type, index_config, start_time)

    def build_from_embeddings(
            self,
            embedding_path: str,
            output_db: str,
            file_extension: Optional[str] = None,
            use_gpu_index: bool = False,
            index_type: IndexType = IndexType.auto,
            index_config: Optional[IndexConfig] = None,
    ):
        """Build a FAISS database from a directory or file of pre-computed embeddings.

        Supports ``.pt``, ``.csv``, and ``.parquet`` inputs via
        :func:`foldmatch.search.embedding_computer.stream_embeddings`. No
        inference is performed; ``tmp_dir`` is not required.
        """
        from foldmatch.search.embedding_search import stream_embeddings
        logging.info("Building FAISS database from pre-computed embeddings")
        start_time = time.time()
        batches = stream_embeddings(embedding_path, file_extension)
        self._create(output_db, batches, use_gpu_index, index_type, index_config, start_time)

    def build_from_fasta(
            self,
            fasta_file: str,
            output_db: str,
            min_res_n: int = 0,
            use_gpu_index: bool = False,
            batch_size: int = 1,
            num_workers: int = 0,
            num_nodes: int = 1,
            devices='auto',
            strategy='auto',
            index_type: IndexType = IndexType.auto,
            index_config: Optional[IndexConfig] = None,
    ):
        """Build a FAISS database from protein sequences in a FASTA file."""
        logging.info("Building embeddings and FAISS database from FASTA")
        start_time = time.time()
        batches = self.computer.compute_from_fasta(
            fasta_file=fasta_file,
            min_res_n=min_res_n,
            batch_size=batch_size,
            num_workers=num_workers,
            num_nodes=num_nodes,
            devices=devices,
            strategy=strategy,
        )
        self._create(output_db, batches, use_gpu_index, index_type, index_config, start_time)

    def _create(self, output_db, batches, use_gpu_index, index_type, index_config, compute_start):
        db_dir, index_name, output_db = _parse_output_db(output_db)
        if is_rank_zero():
            embeddings_time = time.time() - compute_start
            logging.info(f"Creating embeddings completed in {embeddings_time:.2f} seconds")

            start_time = time.time()
            db = FaissEmbeddingDatabase(db_path=str(db_dir), index_name=index_name)
            db.create_database(
                batches=batches,
                index_type=index_type,
                index_config=index_config,
                use_gpu=use_gpu_index,
            )
            database_time = time.time() - start_time
            logging.info(f"Creating database completed in {database_time:.2f} seconds")

            logging.info("Database build complete!")
            logging.info(f"Database location: {output_db}")
            logging.info(f"Total embeddings: {len(db.chain_ids)}")

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
