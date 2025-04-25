import pandas as pd
from torch.utils.data import Dataset, DataLoader

from rcsb_embedding_model.types.api_types import SrcLocation, SrcFormat
from rcsb_embedding_model.utils.data import stringio_from_url
from rcsb_embedding_model.utils.structure_parser import get_structure_from_src, get_protein_chains


class AssemblyEmbeddingFromCsv(Dataset):
    STREAM_ATTR = 'stream'
    ASSEMBLY_ATTR = 'assembly_id'
    NAME_ATTR = 'name'

    COLUMNS = [STREAM_ATTR, ASSEMBLY_ATTR, NAME_ATTR]

    def __init__(
            self,
            csv_file,
            res_embedding_location,
            src_location=SrcLocation.local,
            src_format=SrcFormat.mmcif,
            min_res_n=0
    ):
        super().__init__()
        self.res_embedding_location = res_embedding_location
        self.src_location = src_location
        self.src_format = src_format
        self.min_res_n = min_res_n
        self.data = pd.DataFrame()
        self.__load_stream(csv_file)

    def __load_stream(self, stream_list):
        self.data = pd.read_csv(
            stream_list,
            header=None,
            index_col=None,
            dtype=str,
            names=AssemblyEmbeddingFromCsv.COLUMNS
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_structure = self.data.loc[idx, AssemblyEmbeddingFromCsv.STREAM_ATTR]
        assembly_id = self.data.loc[idx, AssemblyEmbeddingFromCsv.ASSEMBLY_ATTR]
        name = self.data.loc[idx, AssemblyEmbeddingFromCsv.NAME_ATTR]

        structure = get_structure_from_src(
            src_structure=src_structure if self.src_location == SrcLocation.local else stringio_from_url(src_structure),
            src_format=self.src_format,
            assembly_id=assembly_id
        )

        return get_protein_chains(structure, self.min_res_n), name


if __name__ == "__main__":

    dataset = AssemblyEmbeddingFromCsv(
        csv_file="/Users/joan/tmp/assembly-test.csv",
        res_embedding_location="/Users/joan/tmp",
        src_location=SrcLocation.local,
        src_format=SrcFormat.mmcif
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=lambda _: _
    )

    for _batch in dataloader:
        print(_batch)
