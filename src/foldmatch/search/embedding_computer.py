import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import torch.distributed as dist
import os
from secrets import token_hex

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

    def __init__(self, tmp_dir: str, accelerator: Accelerator = 'auto'):
        self.tmp_dir = tmp_dir
        self.out_name = f"emb_{_consolidate_id()}"
        self.accelerator = accelerator

    def compute_from_structures(
            self,
            structure_dir: str,
            structure_format: StructureFormat = StructureFormat.mmcif,
            min_res: int = 10,
            granularity: Granularity = 'chain',
            file_extension: Optional[str] = None,
            batch_size: int = 1,
            num_workers: int = 0,
            num_nodes: int = 1,
            devices='auto',
            strategy='auto',
    ) -> tuple[list, list]:
        """Compute chain or assembly embeddings from a directory of structure files."""
        if file_extension is None:
            file_extension = '.cif' if structure_format == StructureFormat.mmcif else '.pdb'

        structure_dir = Path(structure_dir)
        if not structure_dir.exists():
            raise ValueError(f"Structure directory does not exist: {structure_dir}")

        logging.info(f"Listing structure files from: {structure_dir}")
        structure_files = list(structure_dir.glob(f"*{file_extension}"))
        if not structure_files:
            raise ValueError(
                f"No structure files found with extension {file_extension} in {structure_dir}"
            )


        if granularity == 'chain':
            logging.info(f"Listing residue embedding files from: {self.tmp_dir}")
            chain_predict(
                src_stream=[
                    (str_file.stem, str_file, str_file.stem)
                    for str_file in structure_files
                ],
                src_location=SrcLocation.stream,
                min_res_n=min_res,
                out_path=self.tmp_dir,
                out_name=self.out_name,
                accelerator=self.accelerator,
                batch_size=batch_size,
                num_workers=num_workers,
                num_nodes=num_nodes,
                devices=devices,
                strategy=strategy,
                out_format=OutFormat.parquet,
            )
        else:
            assembly_predict(
                src_stream=[
                    (str_file.stem, str_file, str_file.stem)
                    for str_file in structure_files
                ],
                src_location=SrcLocation.stream,
                out_path=self.tmp_dir,
                out_name=self.out_name,
                accelerator=self.accelerator,
                num_workers=num_workers,
                num_nodes=num_nodes,
                devices=devices,
                strategy=strategy,
                out_format=OutFormat.parquet,
            )
        if _is_distributed():
            dist.barrier()
        return self._load_chain_tensors()

    def compute_from_fasta(
            self,
            fasta_file: str,
            min_res_n: int = 0,
            batch_size: int = 1,
            num_workers: int = 0,
            num_nodes: int = 1,
            devices='auto',
            strategy='auto',
    ) -> tuple[list, list]:
        """Compute chain embeddings from protein sequences in a FASTA file."""

        chain_predict(
            src_stream=fasta_file,
            src_from=SrcEsmFrom.fasta,
            min_res_n=min_res_n,
            batch_size=batch_size,
            num_workers=num_workers,
            num_nodes=num_nodes,
            accelerator=self.accelerator,
            devices=devices,
            out_format=OutFormat.parquet,
            out_path=self.tmp_dir,
            out_name=self.out_name,
            strategy=strategy,
        )

        if _is_distributed():
            dist.barrier()
        return self._load_chain_tensors()

    def _load_chain_tensors(self) -> tuple[list, list]:
        """Load chain embeddings from the per-rank Parquet shards. Non-rank-0 ranks return ([], [])."""
        if not _is_rank_zero():
            return [], []
        parquet_files = sorted(Path(self.tmp_dir).glob(f"{self.out_name}-*.parquet"))
        logging.info(f"Loading embeddings from {len(parquet_files)} Parquet shard(s) in: {self.tmp_dir}")
        names: list = []
        embeddings: list = []
        for parquet_file in parquet_files:
            logging.info(f"    Parquet file {parquet_file}")
            df = pd.read_parquet(parquet_file)
            names.extend(df['id'].tolist())
            embeddings.extend(torch.tensor(emb, dtype=torch.float32) for emb in df['embedding'])
        return names, embeddings


def _is_distributed():
    """Check if the current process is running in distributed mode."""
    return dist.is_available() and dist.is_initialized()

def _is_rank_zero():
    """Check if the current process is rank zero in distributed training."""
    return not _is_distributed() or dist.get_rank() == 0

def _get_rank():
    if _is_distributed():
        return dist.get_rank()
    return 0

def _consolidate_id():
    return os.environ.get('SLURM_JOB_ID', token_hex(16))