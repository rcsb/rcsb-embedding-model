import argparse

import torch
from biotite.structure import chain_iter
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, SamplingConfig
from esm.utils.constants.models import ESM3_OPEN_SMALL
from esm.utils.structure.protein_chain import ProteinChain
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from rcsb_embedding_model.types.api_types import StructureFormat, StructureLocation, SrcLocation
from rcsb_embedding_model.utils.data import stringio_from_url
from rcsb_embedding_model.utils.structure_parser import rename_atom_ch
from rcsb_embedding_model.utils.structure_provider import StructureProvider


class EsmProtFromChain(Dataset):

    STREAM_NAME_ATTR = 'stream_name'
    STREAM_ATTR = 'stream'
    CH_ATTR = 'chain_id'
    ITEM_NAME_ATTR = 'item_name'

    COLUMNS = [STREAM_NAME_ATTR, STREAM_ATTR, CH_ATTR, ITEM_NAME_ATTR]

    def __init__(
        self,
        src_stream,
        src_location=SrcLocation.local,
        structure_location=StructureLocation.local,
        structure_format=StructureFormat.mmcif,
        structure_provider=StructureProvider()
    ):
        super().__init__()
        self.__structure_provider = structure_provider
        self.src_location = src_location
        self.structure_location = structure_location
        self.structure_format = structure_format
        self.data = pd.DataFrame()
        self.__load_stream(src_stream)

    def __load_stream(self, src_stream):
        self.data = pd.DataFrame(
            src_stream,
            dtype=str,
            columns=EsmProtFromChain.COLUMNS
        ) if self.src_location == SrcLocation.stream else pd.read_csv(
            src_stream,
            header=None,
            index_col=None,
            dtype=str,
            names=EsmProtFromChain.COLUMNS
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_name = self.data.loc[idx, EsmProtFromChain.STREAM_NAME_ATTR]
        src_structure = self.data.loc[idx, EsmProtFromChain.STREAM_ATTR]
        chain_id = self.data.loc[idx, EsmProtFromChain.CH_ATTR]
        item_name = self.data.loc[idx, EsmProtFromChain.ITEM_NAME_ATTR]
        structure = self.__structure_provider.get_structure(
            src_name=src_name,
            src_structure=stringio_from_url(src_structure) if self.structure_location == StructureLocation.remote else src_structure,
            structure_format=self.structure_format,
            chain_id=chain_id
        )
        for atom_ch in chain_iter(structure):
            protein_chain = ProteinChain.from_atomarray(rename_atom_ch(atom_ch))
            return ESMProtein.from_protein_chain(protein_chain), item_name


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--file_list', type=argparse.FileType('r'), required=True)
    args = parser.parse_args()

    dataset = EsmProtFromChain(
        args.file_list
    )

    esm3 = ESM3.from_pretrained(
        ESM3_OPEN_SMALL,
        torch.device("cpu")
    )

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=lambda _: _
    )

    for _batch in dataloader:
        for esm_prot, prot_name in _batch:
            protein_tensor = esm3.encode(esm_prot)
            embeddings = esm3.forward_and_sample(
                protein_tensor, SamplingConfig(return_per_residue_embeddings=True)
            ).per_residue_embedding
            print(prot_name, embeddings.shape)
