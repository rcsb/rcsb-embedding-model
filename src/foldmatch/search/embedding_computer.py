import logging
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist

from foldmatch.inference.esm_inference import predict as esm_predict
from foldmatch.inference.chain_inference import predict as chain_predict
from foldmatch.inference.assembly_inferece import predict as assembly_predict
from foldmatch.inference.sequence_inference import predict as sequence_predict
from foldmatch.types.api_types import (
    StructureFormat,
    SrcLocation,
    SrcProteinFrom,
    SrcAssemblyFrom,
    SrcTensorFrom,
    OutFormat,
    Accelerator,
    Granularity,
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
        run_dir = tempfile.mkdtemp(prefix="run_", dir=tmp_dir)
        self.tmp_res_dir = Path(run_dir) / "res"
        self.tmp_res_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_ch_dir = Path(run_dir) / "ch"
        self.tmp_ch_dir.mkdir(parents=True, exist_ok=True)
        self.accelerator = accelerator

    def compute_from_structures(
            self,
            structure_dir: str,
            structure_format: StructureFormat = StructureFormat.mmcif,
            min_res: int = 10,
            granularity: Granularity = 'chain',
            file_extension: Optional[str] = None,
            batch_size_res: int = 1,
            num_workers_res: int = 0,
            num_nodes_res: int = 1,
            batch_size_chain: int = 1,
            num_workers_chain: int = 0,
            num_nodes_chain: int = 1,
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

        esm_predict(
            src_stream=[
                (str_file.stem, str_file, str_file.stem)
                for str_file in structure_files
            ],
            src_location=SrcLocation.stream,
            src_from=SrcProteinFrom.structure,
            structure_format=structure_format,
            min_res_n=min_res,
            out_path=self.tmp_res_dir,
            accelerator=self.accelerator,
            batch_size=batch_size_res,
            num_workers=num_workers_res,
            num_nodes=num_nodes_res,
            devices=devices,
            strategy=strategy,
            write_tensor=True,
        )
        self._consolidate_after_residue()

        if granularity == 'chain':
            logging.info(f"Listing residue embedding files from: {self.tmp_res_dir}")
            esm_embedding_files = list(self.tmp_res_dir.glob("*pt"))
            chain_predict(
                src_stream=[(f, f.stem) for f in esm_embedding_files],
                src_location=SrcLocation.stream,
                out_path=self.tmp_ch_dir,
                accelerator=self.accelerator,
                batch_size=batch_size_chain,
                num_workers=num_workers_chain,
                num_nodes=num_nodes_chain,
                devices=devices,
                strategy=strategy,
                write_tensor=True,
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
                write_tensor=True,
            )
        if _is_distributed():
            dist.barrier()
        return self._load_chain_tensors()

    def compute_from_fasta(
            self,
            fasta_file: str,
            min_res_n: int = 0,
            batch_size_res: int = 1,
            num_workers_res: int = 0,
            num_nodes_res: int = 1,
            batch_size_chain: int = 1,
            num_workers_chain: int = 0,
            num_nodes_chain: int = 1,
            devices='auto',
            strategy='auto',
    ) -> tuple[list, list]:
        """Compute chain embeddings from protein sequences in a FASTA file."""
        from foldmatch.cli.sequence_embedding import scan_fasta_sequences

        sequence_predict(
            fasta_file=fasta_file,
            min_res_n=min_res_n,
            batch_size=batch_size_res,
            num_workers=num_workers_res,
            num_nodes=num_nodes_res,
            accelerator=self.accelerator,
            devices=devices,
            out_format=OutFormat.separated,
            out_path=self.tmp_res_dir,
            strategy=strategy,
            write_tensor=True,
        )
        self._consolidate_after_residue()

        src_stream = scan_fasta_sequences(fasta_file, str(self.tmp_res_dir))
        chain_predict(
            src_stream=src_stream,
            src_location=SrcLocation.stream,
            src_from=SrcTensorFrom.file,
            out_path=self.tmp_ch_dir,
            accelerator=self.accelerator,
            batch_size=batch_size_chain,
            num_workers=num_workers_chain,
            num_nodes=num_nodes_chain,
            devices=devices,
            strategy=strategy,
            write_tensor=True,
        )
        if _is_distributed():
            dist.barrier()
        return self._load_chain_tensors()

    def _consolidate_after_residue(self):
        if _is_distributed():
            dist.barrier()
            self.tmp_res_dir, self.tmp_ch_dir = _consolidate_run_dirs(
                self.tmp_res_dir, self.tmp_ch_dir
            )
            dist.barrier()

    def _load_chain_tensors(self) -> tuple[list, list]:
        logging.info(f"Loading embedding tensors from: {self.tmp_ch_dir}")
        tensor_files = [f for f in self.tmp_ch_dir.iterdir() if f.is_file()]
        names = [f.stem for f in tensor_files]
        embeddings = [torch.load(f) for f in tensor_files]
        return names, embeddings


def _is_distributed():
    """Check if the current process is running in distributed mode."""
    return dist.is_available() and dist.is_initialized()


def _is_rank_zero():
    """Check if the current process is rank zero in distributed training."""
    return not _is_distributed() or dist.get_rank() == 0


def _consolidate_run_dirs(local_res_dir: Path, local_ch_dir: Path) -> tuple[Path, Path]:
    """Unify the per-rank residue and chain temp dirs onto rank 0's paths.

    Each rank ran ``mkdtemp`` independently in ``__init__``, so residue
    embeddings were just written to a different directory on every rank.
    Broadcast rank 0's paths to everyone, move per-rank residue files into
    the canonical residue dir, and remove the now-empty per-rank scratch
    dirs. The returned paths are identical on every rank, so subsequent
    inference and tensor loads see one shared location.

    Must be called with the process group already initialized.
    """
    rank = dist.get_rank()

    res_payload = [str(local_res_dir)] if rank == 0 else [None]
    dist.broadcast_object_list(res_payload, src=0)
    canonical_res = Path(res_payload[0])

    ch_payload = [str(local_ch_dir)] if rank == 0 else [None]
    dist.broadcast_object_list(ch_payload, src=0)
    canonical_ch = Path(ch_payload[0])

    if rank != 0:
        for f in local_res_dir.iterdir():
            if f.is_file():
                shutil.move(str(f), str(canonical_res / f.name))
        # Per-rank res/ and ch/ live under the same parent run dir from mkdtemp.
        local_run_dir = local_res_dir.parent
        try:
            shutil.rmtree(local_run_dir)
        except OSError as e:
            logger.warning(f"Failed to remove per-rank scratch dir {local_run_dir}: {e}")

    return canonical_res, canonical_ch
