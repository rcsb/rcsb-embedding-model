import logging
import torch
from torch.utils.data import DataLoader
from lightning import Trainer

from foldmatch.dataset.esm_assembly_from_structure import EsmAssemblyFromStructure
from foldmatch.modules.assembly_complete_module import AssemblyCompleteModule
from foldmatch.types.api_types import StructureFormat, Accelerator, Devices, Strategy, OptionalPath, \
    FileOrStreamTuple, SrcLocation, OutFormat
from foldmatch.utils.data import identity_collate
from foldmatch.utils.model import get_residue_model, get_aggregator_model
from foldmatch.writer.batch_writer import ParquetBatchWriter, JsonStorage, TensorBatchWriter, CsvBatchWriter


def predict(
        src_stream: FileOrStreamTuple,
        src_location: SrcLocation = SrcLocation.file,
        structure_format: StructureFormat = StructureFormat.mmcif,
        min_res_n: int = 0,
        max_res_n: int = 0,
        batch_size: int = 1,
        num_workers: int = 0,
        num_nodes: int = 1,
        accelerator: Accelerator = 'auto',
        devices: Devices = 'auto',
        strategy: Strategy = 'auto',
        out_format: OutFormat = OutFormat.csv,
        out_name: str = 'inference',
        out_path: OptionalPath = None,
        return_predictions: bool = True,
):
    logger = logging.getLogger(__name__)

    inference_set = EsmAssemblyFromStructure(
        src_stream=src_stream,
        src_location=src_location,
        structure_format=structure_format,
        min_res_n=min_res_n
    )

    logger.info("assembly-inference set running as iterator")

    inference_dataloader = DataLoader(
        dataset=inference_set,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=identity_collate
    )

    logger.info("Loading rcsb-esm + rcsb-aggregator module")
    res_model = get_residue_model(
        device=torch.device("cpu")
    )
    aggregator_model = get_aggregator_model(
        device=torch.device("cpu")
    )
    module = AssemblyCompleteModule(
        res_model=res_model,
        aggregator_model=aggregator_model,
        max_res_n=max_res_n
    )
    logger.info("rcsb-esm + rcsb-aggregator module ready")

    if out_path is not None:
        if out_format == OutFormat.parquet:
            inference_writer = ParquetBatchWriter(out_path, out_name)
        elif out_format == OutFormat.json:
            inference_writer = JsonStorage(out_path, out_name)
        elif out_format == OutFormat.pt:
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

    logger.info("assembly-inference starts")
    prediction = trainer.predict(
        module,
        inference_dataloader,
        return_predictions=return_predictions,
    )

    return prediction
