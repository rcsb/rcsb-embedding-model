import logging
from pathlib import Path
from typing import List

import torch
import pyarrow as pa
import pyarrow.parquet as pq

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class ResidueEmbeddingFromParquet(Dataset):

    def __init__(
            self,
            parquet_path: Path | List[Path],
            embedding_dim: int
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        if isinstance(parquet_path, (str, Path)):
            paths = [parquet_path]
        else:
            paths = parquet_path
        logger.info(f"Loading {len(paths)} parquet file(s): {[str(p) for p in paths]}")
        self.table = pa.concat_tables([pq.read_table(p) for p in paths])

    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx):
        item_id = self.table.column('id')[idx].as_py()
        embedding_flat = self.table.column('embedding')[idx].as_py()
        tensor = torch.tensor(embedding_flat, dtype=torch.float32).reshape(-1, self.embedding_dim)
        return tensor, item_id
