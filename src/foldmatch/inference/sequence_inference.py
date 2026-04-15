import logging
import torch
from torch.utils.data import DataLoader
from lightning import Trainer

from foldmatch.dataset.esm_prot_from_fasta import EsmProtFromFasta
from foldmatch.modules.esm_module import EsmModule
from foldmatch.types.api_types import Accelerator, Devices, Strategy, OptionalPath, OutFormat
from foldmatch.utils.data import identity_collate
from foldmatch.utils.model import get_residue_model
from foldmatch.writer.batch_writer import TensorBatchWriter, CsvBatchWriter, JsonStorage


def predict(
        fasta_file: str,
        min_res_n: int = 0,
        batch_size: int = 1,
        num_workers: int = 0,
        num_nodes: int = 1,
        accelerator: Accelerator = 'auto',
        devices: Devices = 'auto',
        strategy: Strategy = 'auto',
        out_format: OutFormat = OutFormat.separated,
        out_name: str = 'inference',
        out_path: OptionalPath = None,
        write_tensor: bool = False
):
    logger = logging.getLogger(__name__)

    inference_set = EsmProtFromFasta(
        fasta_file=fasta_file,
        min_res_n=min_res_n
    )
    logger.info(f"sequence-inference set contains {len(inference_set)} samples")

    inference_dataloader = DataLoader(
        dataset=inference_set,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=identity_collate
    )

    logger.info(f"Loading rcsb-esm module")
    esm_model = get_residue_model(
        device=torch.device("cpu")
    )
    module = EsmModule(
        model=esm_model
    )
    logger.info(f"rcsb-esm module ready")

    if out_path is not None:
        if out_format == OutFormat.grouped:
            inference_writer = JsonStorage(out_path, out_name)
        elif write_tensor:
            inference_writer = TensorBatchWriter(out_path)
        else:
            inference_writer = CsvBatchWriter(out_path)
    else:
        inference_writer = None
    trainer = Trainer(
        callbacks=[inference_writer] if inference_writer is not None else None,
        num_nodes=num_nodes,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        logger=False
    )

    logger.info(f"sequence-inference starts")
    prediction = trainer.predict(
        module,
        inference_dataloader
    )

    return prediction
