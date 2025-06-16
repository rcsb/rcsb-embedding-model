import pandas as pd
import torch
from torch.utils.data import Dataset

from rcsb_embedding_model.types.api_types import SrcLocation


class ResidueEmbeddingFromTensorFile(Dataset):

    FILE_ATTR = 'file'
    ITEM_NAME_ATTR = 'item_name'

    COLUMNS = [FILE_ATTR, ITEM_NAME_ATTR]

    def __init__(
            self,
            src_stream,
            src_location=SrcLocation.file
    ):
        super().__init__()
        self.src_location = src_location
        self.data = pd.DataFrame()
        self.__load_stream(src_stream)

    def __load_stream(self, src_stream):
        self.data = pd.DataFrame(
            src_stream,
            dtype=str,
            columns=ResidueEmbeddingFromTensorFile.COLUMNS
        ) if self.src_location == SrcLocation.stream else pd.read_csv(
            src_stream,
            header=None,
            index_col=None,
            names=ResidueEmbeddingFromTensorFile.COLUMNS
        )
        self.data = self.data.sort_values(by=self.data.columns[0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        embedding_src = self.data.iloc[idx][ResidueEmbeddingFromTensorFile.FILE_ATTR]
        item_name = self.data.iloc[idx][ResidueEmbeddingFromTensorFile.ITEM_NAME_ATTR]
        return torch.load(embedding_src, map_location=torch.device('cpu')), item_name
