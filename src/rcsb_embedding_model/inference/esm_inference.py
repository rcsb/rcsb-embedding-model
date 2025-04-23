import argparse

from torch.utils.data import DataLoader
from lightning import Trainer

from rcsb_embedding_model.dataset.esm_prot_from_stream_list import EsmProtFromStreamList
from rcsb_embedding_model.modules.esm_module import EsmModule
from rcsb_embedding_model.types.api_types import StreamList, SrcFormat, Accelerator, Devices, OptionalPath
from rcsb_embedding_model.writer.batch_writer import TensorBatchWriter


def predict(
        stream_list: StreamList,
        src_format: SrcFormat = SrcFormat.mmcif,
        batch_size: int = 1,
        num_workers: int = 0,
        num_nodes: int = 1,
        accelerator: Accelerator = Accelerator.auto,
        devices: Devices = 'auto',
        out_path: OptionalPath = None
):

    inference_set = EsmProtFromStreamList(
        stream_list,
        src_format=src_format
    )

    inference_dataloader = DataLoader(
        dataset=inference_set,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=lambda _: _
    )

    module = EsmModule()
    inference_writer = TensorBatchWriter(out_path) if out_path is not None else None
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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--file_list', type=argparse.FileType('r'), required=True)
    parser.add_argument('--out_path', type=str, required=True)
    args = parser.parse_args()

    preds = predict(
        stream_list=args.file_list,
        batch_size=2,
        out_path=args.out_path,
        accelerator=Accelerator.cpu
    )
