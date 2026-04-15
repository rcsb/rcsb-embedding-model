import logging
import torch
from torch.utils.data import DataLoader
from lightning import Trainer

from foldmatch.dataset.residue_embedding_from_structure import ResidueEmbeddingFromStructure
from foldmatch.dataset.residue_embedding_from_tensor_file import ResidueEmbeddingFromTensorFile
from foldmatch.modules.chain_module import ChainModule
from foldmatch.types.api_types import Accelerator, Devices, Strategy, OptionalPath, FileOrStreamTuple, SrcLocation, \
    SrcTensorFrom, StructureFormat, OutFormat
from foldmatch.utils.data import collate_seq_embeddings, collate_embeddings
from foldmatch.utils.model import get_aggregator_model
from foldmatch.writer.batch_writer import CsvBatchWriter, TensorBatchWriter, JsonStorage


def predict(
        src_stream: FileOrStreamTuple,
        res_embedding_location: OptionalPath = None,
        src_location: SrcLocation = SrcLocation.file,
        src_from: SrcTensorFrom = SrcTensorFrom.file,
        structure_format: StructureFormat = StructureFormat.mmcif,
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
        inference_set=None,
        write_tensor: bool = False
):
    logger = logging.getLogger(__name__)

    if inference_set is None:
        inference_set = ResidueEmbeddingFromTensorFile(
            src_stream=src_stream,
            src_location=src_location
        ) if src_from == SrcTensorFrom.file else ResidueEmbeddingFromStructure(
            src_stream=src_stream,
            res_embedding_location=res_embedding_location,
            src_location=src_location,
            structure_format=structure_format,
            min_res_n=min_res_n
        )
    logger.info(f"chain-inference set contains {len(inference_set)} samples")

    inference_dataloader = DataLoader(
        dataset=inference_set,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_embeddings
    )

    logger.info(f"Loading rcsb-aggregator module")
    aggregator_model = get_aggregator_model(
        device=torch.device("cpu")
    )
    module = ChainModule(
        model=aggregator_model
    )
    logger.info(f"rcsb-aggregator module ready")

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

    logger.info(f"chain-inference starts")
    prediction = trainer.predict(
        module,
        inference_dataloader
    )

    return prediction
