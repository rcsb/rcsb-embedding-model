import logging
import torch
from torch.utils.data import DataLoader
from lightning import Trainer

from foldmatch.dataset.esm_prot_from_structure import EsmProtFromStructure
from foldmatch.dataset.esm_prot_from_chain import EsmProtFromChain
from foldmatch.modules.structure_module import StructureModule
from foldmatch.types.api_types import StructureFormat, Accelerator, Devices, Strategy, OptionalPath, \
    SrcProteinFrom, FileOrStreamTuple, SrcLocation
from foldmatch.utils.data import identity_collate
from foldmatch.utils.model import get_residue_model, get_aggregator_model
from foldmatch.writer.batch_writer import JsonStorage


def predict(
        src_stream: FileOrStreamTuple,
        src_location: SrcLocation = SrcLocation.file,
        src_from: SrcProteinFrom = SrcProteinFrom.chain,
        structure_format: StructureFormat = StructureFormat.mmcif,
        min_res_n: int = 0,
        batch_size: int = 1,
        num_workers: int = 0,
        num_nodes: int = 1,
        accelerator: Accelerator = 'auto',
        devices: Devices = 'auto',
        strategy: Strategy = 'auto',
        out_name: str = 'inference',
        out_path: OptionalPath = None
):
    logger = logging.getLogger(__name__)

    inference_set = EsmProtFromChain(
        src_stream=src_stream,
        src_location=src_location,
        structure_format=structure_format
    ) if src_from == SrcProteinFrom.chain else EsmProtFromStructure(
        src_stream=src_stream,
        src_location=src_location,
        structure_format=structure_format,
        min_res_n=min_res_n
    )
    logger.info(f"structure-inference set running as iterator")

    inference_dataloader = DataLoader(
        dataset=inference_set,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=identity_collate
    )

    logger.info(f"Loading rcsb-esm + rcsb-aggregator module")
    res_model = get_residue_model(
        device=torch.device("cpu")
    )
    aggregator_model = get_aggregator_model(
        device=torch.device("cpu")
    )
    module = StructureModule(
        res_model=res_model,
        aggregator_model=aggregator_model
    )
    logger.info(f"rcsb-esm + rcsb-aggregator module ready")

    inference_writer = JsonStorage(out_path, out_name) if out_path is not None and out_name is not None else None
    trainer = Trainer(
        callbacks=[inference_writer] if inference_writer is not None else None,
        num_nodes=num_nodes,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        logger=False
    )

    logger.info(f"structure-inference starts")
    prediction = trainer.predict(
        module,
        inference_dataloader
    )

    return prediction
