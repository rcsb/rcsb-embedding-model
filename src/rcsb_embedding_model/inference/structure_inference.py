from torch.utils.data import DataLoader
from lightning import Trainer

from rcsb_embedding_model.dataset.esm_prot_from_structure import EsmProtFromStructure
from rcsb_embedding_model.dataset.esm_prot_from_chain import EsmProtFromChain
from rcsb_embedding_model.modules.structure_module import StructureModule
from rcsb_embedding_model.types.api_types import StructureFormat, Accelerator, Devices, OptionalPath, StructureLocation, SrcProteinFrom, FileOrStreamTuple, SrcLocation
from rcsb_embedding_model.writer.batch_writer import DataFrameStorage


def predict(
        src_stream: FileOrStreamTuple,
        src_location: SrcLocation = SrcLocation.local,
        src_from: SrcProteinFrom = SrcProteinFrom.chain,
        structure_location: StructureLocation = StructureLocation.local,
        structure_format: StructureFormat = StructureFormat.mmcif,
        min_res_n: int = 0,
        batch_size: int = 1,
        num_workers: int = 0,
        num_nodes: int = 1,
        accelerator: Accelerator = Accelerator.auto,
        devices: Devices = 'auto',
        out_path: OptionalPath = None,
        out_df_name: str = None
):

    inference_set = EsmProtFromChain(
        src_stream=src_stream,
        src_location=src_location,
        structure_location=structure_location,
        structure_format=structure_format
    ) if src_from == SrcProteinFrom.chain else EsmProtFromStructure(
        src_stream=src_stream,
        src_location=src_location,
        structure_location=structure_location,
        structure_format=structure_format,
        min_res_n=min_res_n
    )

    inference_dataloader = DataLoader(
        dataset=inference_set,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=lambda _: _
    )

    module = StructureModule()
    inference_writer = DataFrameStorage(out_path, out_df_name) if out_path is not None and out_df_name is not None else None
    trainer = Trainer(
        callbacks=[inference_writer] if inference_writer is not None else None,
        num_nodes=num_nodes,
        accelerator=accelerator,
        devices=devices
    )

    prediction = trainer.predict(
        module,
        inference_dataloader
    )

    return prediction
