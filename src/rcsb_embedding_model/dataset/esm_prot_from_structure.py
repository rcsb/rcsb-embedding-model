import argparse

import pandas as pd
from torch.utils.data import DataLoader

from rcsb_embedding_model.dataset.esm_prot_from_chain import EsmProtFromChain
from rcsb_embedding_model.types.api_types import SrcLocation, SrcFormat
from rcsb_embedding_model.utils.data import stringio_from_url
from rcsb_embedding_model.utils.structure_parser import get_protein_chains
from rcsb_embedding_model.utils.structure_provider import StructureProvider


class EsmProtFromStructure(EsmProtFromChain):

    STREAM_NAME_ATTR = 'stream_name'
    STREAM_ATTR = 'stream'
    ITEM_NAME_ATTR = 'item_name'

    COLUMNS = [STREAM_NAME_ATTR, STREAM_ATTR, ITEM_NAME_ATTR]

    def __init__(
            self,
            src_stream,
            src_location=SrcLocation.local,
            src_format=SrcFormat.mmcif,
            min_res_n=0,
            structure_provider=StructureProvider()
    ):
        self.min_res_n = min_res_n
        self.src_location = src_location
        self.src_format = src_format
        self.__structure_provider = structure_provider
        super().__init__(
            src_stream=self.__get_chains(src_stream),
            src_location=SrcLocation.stream,
            src_format=src_format
        )

    def __get_chains(self, src_stream):
        chains = []
        for idx, row in (pd.DataFrame(
            src_stream,
            dtype=str,
            columns=self.COLUMNS
        ) if self.src_location == SrcLocation.stream else pd.read_csv(
            src_stream,
            header=None,
            index_col=None,
            dtype=str,
            names=EsmProtFromStructure.COLUMNS
        )).iterrows():
            src_name = row[EsmProtFromStructure.STREAM_NAME_ATTR]
            src_structure = row[EsmProtFromStructure.STREAM_ATTR]
            item_name = row[EsmProtFromStructure.ITEM_NAME_ATTR]
            structure = self.__structure_provider.get_structure(
                src_name=src_name,
                src_structure=stringio_from_url(src_structure) if self.src_location == SrcLocation.remote else src_structure,
                src_format=self.src_format
            )
            for ch in get_protein_chains(structure, self.min_res_n):
                chains.append((src_name, src_structure, ch, f"{item_name}.{ch}"))
        return tuple(chains)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--file_list', type=argparse.FileType('r'), required=True)
    args = parser.parse_args()

    dataset = EsmProtFromStructure(
        args.file_list
    )

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=lambda _: _
    )

    for _batch in dataloader:
        for esm_prot, name in _batch:
            print(name)
