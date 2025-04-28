import sys

import pandas as pd
from torch.utils.data import Dataset, DataLoader

from rcsb_embedding_model.types.api_types import SrcLocation, SrcFormat
from rcsb_embedding_model.utils.data import stringio_from_url, concatenate_tensors
from rcsb_embedding_model.utils.structure_parser import get_structure_from_src, get_protein_chains
from rcsb_embedding_model.utils.structure_provider import StructureProvider


class ResidueAssemblyEmbeddingFromTensorFile(Dataset):

    STREAM_NAME_ATTR = 'stream_name'
    STREAM_ATTR = 'stream'
    ASSEMBLY_ATTR = 'assembly_id'
    ITEM_NAME_ATTR = 'item_name'

    COLUMNS = [STREAM_NAME_ATTR, STREAM_ATTR, ASSEMBLY_ATTR, ITEM_NAME_ATTR]

    def __init__(
            self,
            src_stream,
            res_embedding_location,
            src_location=SrcLocation.local,
            src_format=SrcFormat.mmcif,
            min_res_n=0,
            max_res_n=sys.maxsize,
            structure_provider=StructureProvider()
    ):
        super().__init__()
        self.res_embedding_location = res_embedding_location
        self.src_location = src_location
        self.src_format = src_format
        self.min_res_n = min_res_n
        self.max_res_n = max_res_n
        self.data = pd.DataFrame()
        self.__load_stream(src_stream)
        self.__structure_provider = structure_provider

    def __load_stream(self, src_stream):
        self.data = pd.DataFrame(
            src_stream,
            dtype=str,
            columns=ResidueAssemblyEmbeddingFromTensorFile.COLUMNS
        ) if self.src_location == SrcLocation.stream else pd.read_csv(
            src_stream,
            header=None,
            index_col=None,
            dtype=str,
            names=ResidueAssemblyEmbeddingFromTensorFile.COLUMNS
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_name = self.data.loc[idx, ResidueAssemblyEmbeddingFromTensorFile.STREAM_NAME_ATTR]
        src_structure = self.data.loc[idx, ResidueAssemblyEmbeddingFromTensorFile.STREAM_ATTR]
        assembly_id = self.data.loc[idx, ResidueAssemblyEmbeddingFromTensorFile.ASSEMBLY_ATTR]
        item_name = self.data.loc[idx, ResidueAssemblyEmbeddingFromTensorFile.ITEM_NAME_ATTR]

        structure = self.__structure_provider.get_structure(
            src_name=src_name,
            src_structure=stringio_from_url(src_structure) if self.src_location == SrcLocation.remote else src_structure,
            src_format=self.src_format,
            assembly_id=assembly_id
        )
        residue_embedding_files = [
            f"{self.res_embedding_location}/{src_name}.{ch}.pt" for ch in get_protein_chains(structure, self.min_res_n)
        ]
        return concatenate_tensors(residue_embedding_files, self.max_res_n), item_name


if __name__ == "__main__":

    dataset = ResidueAssemblyEmbeddingFromTensorFile(
        src_stream="/Users/joan/tmp/assembly-test.csv",
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
