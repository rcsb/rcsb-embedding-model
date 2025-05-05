
from abc import abstractmethod
from collections import deque
from abc import ABC

import torch
import pandas as pd

from lightning.pytorch.callbacks import BasePredictionWriter


class CoreBatchWriter(BasePredictionWriter, ABC):
    def __init__(
            self,
            output_path,
            postfix,
            write_interval="batch"
    ):
        super().__init__(write_interval)
        self.out_path = output_path
        self.postfix = postfix

    def write_on_batch_end(
            self,
            trainer,
            pl_module,
            prediction,
            batch_indices,
            batch,
            batch_idx,
            dataloader_idx
    ):
        if prediction is None:
            return
        embeddings, dom_ids = prediction
        deque(map(
            self._write_embedding,
            embeddings,
            dom_ids
        ))

    def file_name(self, dom_id):
        return f'{self.out_path}/{dom_id}.{self.postfix}'

    @abstractmethod
    def _write_embedding(self, embedding, dom_id):
        pass


class CsvBatchWriter(CoreBatchWriter, ABC):
    def __init__(
            self,
            output_path,
            postfix="csv",
            write_interval="batch"
    ):
        super().__init__(output_path, postfix, write_interval)

    def _write_embedding(self, embedding, dom_id):
        pd.DataFrame(embedding.to('cpu').numpy()).to_csv(
            self.file_name(dom_id),
            index=False,
            header=False
        )


class TensorBatchWriter(CoreBatchWriter, ABC):
    def __init__(
            self,
            output_path,
            postfix="pt",
            write_interval="batch",
            device="cpu"
    ):
        super().__init__(output_path, postfix, write_interval)
        self.device = device

    def _write_embedding(self, embedding, dom_id):
        torch.save(
            embedding.to(self.device),
            self.file_name(dom_id)
        )


class DataFrameStorage(CoreBatchWriter, ABC):
    def __init__(
            self,
            output_path,
            df_id,
            postfix="pkl",
            write_interval="batch"
    ):
        super().__init__(output_path, postfix, write_interval)
        self.df_id = df_id
        self.embedding = pd.DataFrame(
            data={},
            columns=['id', 'embedding'],
        )

    def _write_embedding(self, embedding, dom_id):
        self.embedding = pd.concat([
            self.embedding,
            pd.DataFrame(
                data={'id': dom_id, 'embedding': [embedding.to('cpu').numpy()]},
                columns=['id', 'embedding'],
            )
        ], ignore_index=True)

    def on_predict_end(self, trainer, pl_module):
        self.embedding.to_pickle(
            f"{self.out_path}/{self.df_id}.pkl.gz",
            compression='gzip'
        )


class JsonStorage(DataFrameStorage, ABC):
    def __init__(
            self,
            output_path,
            df_id,
            postfix="pkl",
            write_interval="batch"
    ):
        super().__init__(output_path, df_id, postfix, write_interval)

    def on_predict_end(self, trainer, pl_module):
        self.embedding.to_json(
            f"{self.out_path}/{self.df_id}.json.gz",
            orient='records',
            compression='gzip'
        )