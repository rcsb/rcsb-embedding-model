import os

import pandas as pd

from rcsb_embedding_model.dataset.residue_embedding_from_tensor_file import ResidueEmbeddingFromTensorFile
from rcsb_embedding_model.types.api_types import SrcLocation, StructureLocation, StructureFormat
from rcsb_embedding_model.utils.data import stringio_from_url
from rcsb_embedding_model.utils.structure_parser import get_protein_chains
from rcsb_embedding_model.utils.structure_provider import StructureProvider


class ResidueEmbeddingFromStructure(ResidueEmbeddingFromTensorFile):

    STREAM_NAME_ATTR = 'stream_name'
    STREAM_ATTR = 'stream'
    ITEM_NAME_ATTR = 'item_name'

    COLUMNS = [STREAM_NAME_ATTR, STREAM_ATTR, ITEM_NAME_ATTR]

    def __init__(
            self,
            src_stream,
            res_embedding_location,
            src_location=SrcLocation.local,
            structure_location=StructureLocation.local,
            structure_format=StructureFormat.mmcif,
            min_res_n=0,
            structure_provider=StructureProvider()
    ):
        if not os.path.isdir(res_embedding_location):
            raise FileNotFoundError(f"Folder {res_embedding_location} does not exist")
        self.res_embedding_location = res_embedding_location
        self.src_location = src_location
        self.structure_location = structure_location
        self.structure_format = structure_format
        self.min_res_n = min_res_n
        self.__structure_provider = structure_provider
        super().__init__(
            src_stream=self.__get_chains(src_stream),
            src_location=SrcLocation.stream
        )

    def __get_chains(self, src_stream):
        chains = []
        for idx, row in (pd.DataFrame(
                src_stream,
                dtype=str,
                columns=ResidueEmbeddingFromStructure.COLUMNS
        ) if self.src_location == SrcLocation.stream else pd.read_csv(
            src_stream,
            header=None,
            index_col=None,
            dtype=str,
            names=ResidueEmbeddingFromStructure.COLUMNS
        )).iterrows():
            src_name = row[ResidueEmbeddingFromStructure.STREAM_NAME_ATTR]
            src_structure = row[ResidueEmbeddingFromStructure.STREAM_ATTR]
            item_name = row[ResidueEmbeddingFromStructure.ITEM_NAME_ATTR]
            structure = self.__structure_provider.get_structure(
                src_name=src_name,
                src_structure=stringio_from_url(src_structure) if self.structure_location == StructureLocation.remote else src_structure,
                structure_format=self.structure_format
            )
            for ch in get_protein_chains(structure, self.min_res_n):
                chains.append((os.path.join(self.res_embedding_location, f"{src_name}.{ch}.pt"), f"{item_name}.{ch}"))
        return tuple(chains)
