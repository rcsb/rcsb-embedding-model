import logging
import os
from pathlib import Path
from secrets import token_hex
from typing import Iterator, Optional

import numpy as np
import pyarrow.parquet as pq

from foldmatch.inference.assembly_complete_inference import predict as assembly_predict
from foldmatch.inference.chain_complete_inference import predict as chain_predict
from foldmatch.types.api_types import (
    StructureFormat,
    SrcLocation,
    OutFormat,
    Accelerator,
    Granularity, SrcEsmFrom,
)

logger = logging.getLogger(__name__)

class EmbeddingComputer:
    """Compute chain or assembly embeddings via a residue->chain inference pipeline.

    Owns the per-run scratch directories and the multi-GPU bookkeeping: each
    rank's __init__ creates its own per-rank scratch dir, then after residue
    inference all ranks consolidate onto rank 0's directory so chain/assembly
    inference and tensor loading see a single shared location.
    """

    def __init__(
            self,
            embedding_folder: str,
    ):
        self.embedding_folder = embedding_folder
        self.out_name = f"emb_{_consolidate_id()}"

    def compute_from_structures(
            self,
            structure_folder: str,
            structure_format: StructureFormat = StructureFormat.mmcif,
            min_res: int = 10,
            granularity: Granularity = 'chain',
            file_extension: Optional[str] = None,
            batch_size: int = 1,
            num_workers: int = 0,
            num_nodes: int = 1,
            accelerator: Accelerator = 'auto',
            devices='auto',
            strategy='auto',
    ):
        """Compute chain or assembly embeddings from a directory of structure files."""
        if file_extension is None:
            file_extension = '.cif' if structure_format == StructureFormat.mmcif else '.pdb'

        structure_folder = Path(structure_folder)
        if not structure_folder.exists():
            raise ValueError(f"Structure directory does not exist: {structure_folder}")

        logging.info(f"Listing structure files from: {structure_folder}")
        structure_files = list(structure_folder.glob(f"*{file_extension}"))
        if not structure_files:
            raise ValueError(
                f"No structure files found with extension {file_extension} in {structure_folder}"
            )

        if granularity == 'chain':
            chain_predict(
                src_stream=[
                    (str_file.stem, str_file, str_file.stem)
                    for str_file in structure_files
                ],
                src_location=SrcLocation.stream,
                structure_format=structure_format,
                min_res_n=min_res,
                out_path=self.embedding_folder,
                out_name=self.out_name,
                accelerator=accelerator,
                batch_size=batch_size,
                num_workers=num_workers,
                num_nodes=num_nodes,
                devices=devices,
                strategy=strategy,
                out_format=OutFormat.parquet,
                return_predictions=False,
            )
        else:
            assembly_predict(
                src_stream=[
                    (str_file.stem, str_file, str_file.stem)
                    for str_file in structure_files
                ],
                src_location=SrcLocation.stream,
                structure_format=structure_format,
                out_path=self.embedding_folder,
                out_name=self.out_name,
                accelerator=accelerator,
                num_workers=num_workers,
                num_nodes=num_nodes,
                devices=devices,
                strategy=strategy,
                out_format=OutFormat.parquet,
                return_predictions=False,
            )

    def compute_from_fasta(
            self,
            fasta_file: str,
            min_res_n: int = 0,
            batch_size: int = 1,
            num_workers: int = 0,
            num_nodes: int = 1,
            devices='auto',
            strategy='auto',
            accelerator: Accelerator = 'auto',
    ):
        """Compute chain embeddings from protein sequences in a FASTA file."""

        chain_predict(
            src_stream=fasta_file,
            src_from=SrcEsmFrom.fasta,
            min_res_n=min_res_n,
            batch_size=batch_size,
            num_workers=num_workers,
            num_nodes=num_nodes,
            accelerator=accelerator,
            devices=devices,
            out_format=OutFormat.parquet,
            out_path=self.embedding_folder,
            out_name=self.out_name,
            strategy=strategy,
            return_predictions=False,
        )

    def get_embedding_batches(self, load_batch_size: int = 4096) -> Iterator[tuple[list[str], np.ndarray]]:
        """Stream (ids, [B, D] float32) batches from all per-rank Parquet shards. Rank-0 only."""

        parquet_files = sorted(Path(self.embedding_folder).glob(f"{self.out_name}-*.parquet"))
        logging.info(f"Streaming embeddings from {len(parquet_files)} Parquet shard(s) in: {self.embedding_folder}")
        for parquet_file in parquet_files:
            logging.info(f"   Parquet file {parquet_file}")
            pf = pq.ParquetFile(parquet_file)
            for record in pf.iter_batches(batch_size=load_batch_size, columns=['id', 'embedding']):
                ids = record.column('id').to_pylist()
                flat = record.column('embedding').values.to_numpy(zero_copy_only=False)
                arr = np.ascontiguousarray(
                    flat.reshape(len(record), -1).astype(np.float32, copy=False)
                )
                yield ids, arr

def _consolidate_id():
    return os.environ.get('SLURM_JOB_ID', token_hex(16))