from torch.utils.data import DataLoader
from lightning import Trainer

from rcsb_embedding_model.dataset.residue_embedding_from_tensor_file import ResidueEmbeddingFromTensorFile
from rcsb_embedding_model.modules.chain_module import ChainModule
from rcsb_embedding_model.types.api_types import Accelerator, Devices, OptionalPath, FileOrStreamTuple, SrcLocation
from rcsb_embedding_model.utils.data import collate_seq_embeddings
from rcsb_embedding_model.writer.batch_writer import CsvBatchWriter


def predict(
        src_stream: FileOrStreamTuple,
        src_location: SrcLocation = SrcLocation.local,
        batch_size: int = 1,
        num_workers: int = 0,
        num_nodes: int = 1,
        accelerator: Accelerator = Accelerator.auto,
        devices: Devices = 'auto',
        out_path: OptionalPath = None,
        inference_set=None
):

    if inference_set is None:
        inference_set = ResidueEmbeddingFromTensorFile(
            src_stream=src_stream,
            src_location=src_location
        )

    inference_dataloader = DataLoader(
        dataset=inference_set,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=lambda emb: (
            collate_seq_embeddings([x for x, z in emb]),
            tuple([z for x, z in emb])
        )
    )

    module = ChainModule()

    inference_writer = CsvBatchWriter(out_path) if out_path is not None else None
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
