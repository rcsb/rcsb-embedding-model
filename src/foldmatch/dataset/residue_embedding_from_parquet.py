import logging

import torch
import pyarrow.parquet as pq

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class ResidueEmbeddingFromParquet(Dataset):

    def __init__(self, parquet_path, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        logger.info(f"Loading parquet file: {parquet_path}")
        self.table = pq.read_table(parquet_path)

    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx):
        item_id = self.table.column('id')[idx].as_py()
        embedding_flat = self.table.column('embedding')[idx].as_py()
        tensor = torch.tensor(embedding_flat, dtype=torch.float32).reshape(-1, self.embedding_dim)
        return tensor, item_id
