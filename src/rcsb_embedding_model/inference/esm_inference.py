from torch.utils.data import DataLoader
from lightning import Trainer

from rcsb_embedding_model.dataset.esm_prot_from_stream_list import EsmProtFromStreamList
from rcsb_embedding_model.modules.esm_module import EsmModule
from rcsb_embedding_model.writer.batch_writer import TensorBatchWriter


def predict(
        stream_list,
        src_format="mmcif",
        batch_size=1,
        num_workers=0,
        num_nodes=1,
        accelerator='cpu',
        devices='auto',
        out_path=None
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
    predict(
        stream_list=[
            ("/Users/joan/tmp/8GYM.cif", "A", "8GYM.A"),
            ("/Users/joan/tmp/2YBB.cif", "A", "2YBB.A")
        ],
        batch_size=2,
        out_path="/Users/joan/tmp"
    )
