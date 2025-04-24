import pandas as pd
import torch
from torch.utils.data import Dataset


class ResidueEmbeddingFromCSV(Dataset):

    STREAM_ATTR = 'stream'
    NAME_ATTR = 'name'

    COLUMNS = [STREAM_ATTR, NAME_ATTR]

    def __init__(self, csv_file):
        super().__init__()
        self.data = pd.DataFrame()
        self.__load_stream(csv_file)

    def __load_stream(self, csv_file):
        self.data = pd.read_csv(
            csv_file,
            header=None,
            index_col=None,
            names=ResidueEmbeddingFromCSV.COLUMNS
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        embedding_src = self.data.loc[idx, ResidueEmbeddingFromCSV.STREAM_ATTR]
        name = self.data.loc[idx, ResidueEmbeddingFromCSV.NAME_ATTR]
        return torch.load(embedding_src, map_location=torch.device('cpu')), name
