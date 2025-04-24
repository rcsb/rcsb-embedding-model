import argparse

import torch
from biotite.structure import chain_iter
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, SamplingConfig
from esm.utils.constants.models import ESM3_OPEN_SMALL
from esm.utils.structure.protein_chain import ProteinChain
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from rcsb_embedding_model.types.api_types import SrcFormat, SrcLocation
from rcsb_embedding_model.utils.data import stringio_from_url
from rcsb_embedding_model.utils.structure_parser import get_structure_from_src


class EsmProtFromCsv(Dataset):

    MIN_RES = 10
    STREAM_ATTR = 'stream'
    CH_ATTR = 'chain_id'
    NAME_ATTR = 'name'

    COLUMNS = [STREAM_ATTR, CH_ATTR, NAME_ATTR]

    def __init__(
        self,
        csv_file,
        src_location=SrcLocation.local,
        src_format=SrcFormat.mmcif,
    ):
        super().__init__()
        self.src_location = src_location
        self.src_format = src_format
        self.data = pd.DataFrame()
        self.__load_stream(csv_file)

    def __load_stream(self, stream_list):
        self.data = pd.read_csv(
            stream_list,
            header=None,
            index_col=None,
            names=EsmProtFromCsv.COLUMNS
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_structure = self.data.loc[idx, EsmProtFromCsv.STREAM_ATTR]
        chain_id = self.data.loc[idx, EsmProtFromCsv.CH_ATTR]
        name = self.data.loc[idx, EsmProtFromCsv.NAME_ATTR]
        structure = get_structure_from_src(
            src_structure=src_structure if self.src_location == SrcLocation.local else stringio_from_url(src_structure),
            src_format=self.src_format,
            chain_id=chain_id
        )
        for atom_ch in chain_iter(structure):
            protein_chain = ProteinChain.from_atomarray(atom_ch)
            return ESMProtein.from_protein_chain(protein_chain), name


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--file_list', type=argparse.FileType('r'), required=True)
    args = parser.parse_args()

    dataset = EsmProtFromCsv(
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
        for esm_prot, name in _batch:
            protein_tensor = esm3.encode(esm_prot)
            embeddings = esm3.forward_and_sample(
                protein_tensor, SamplingConfig(return_per_residue_embeddings=True)
            ).per_residue_embedding
            print(name, embeddings.shape)
