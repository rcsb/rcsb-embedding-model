from torch.utils.data import DataLoader
from lightning import Trainer
from typer import FileText

from rcsb_embedding_model.dataset.esm_prot_from_csv import EsmProtFromCsv
from rcsb_embedding_model.modules.esm_module import EsmModule
from rcsb_embedding_model.types.api_types import SrcFormat, Accelerator, Devices, OptionalPath, SrcLocation
from rcsb_embedding_model.writer.batch_writer import DataFrameStorage


def predict(
        csv_file: FileText,
        src_location: SrcLocation = SrcLocation.local,
        src_format: SrcFormat = SrcFormat.mmcif,
        batch_size: int = 1,
        num_workers: int = 0,
        num_nodes: int = 1,
        accelerator: Accelerator = Accelerator.auto,
        devices: Devices = 'auto',
        out_path: OptionalPath = None,
        out_df_id: str = None
):

    inference_set = EsmProtFromCsv(
        csv_file=csv_file,
        src_location=src_location,
        src_format=src_format
    )

    inference_dataloader = DataLoader(
        dataset=inference_set,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=lambda _: _
    )

    module = EsmModule()
    inference_writer = DataFrameStorage(out_path, out_df_id) if out_path is not None and out_df_id is not None else None
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
