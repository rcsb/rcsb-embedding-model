
import pandas as pd

from rcsb_embedding_model.dataset.esm_prot_from_chain import EsmProtFromChain
from rcsb_embedding_model.dataset.untils import get_structure_location
from rcsb_embedding_model.types.api_types import StructureLocation, StructureFormat, SrcLocation
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
            src_location=SrcLocation.file,
            structure_format=StructureFormat.mmcif,
            min_res_n=0,
            structure_provider=StructureProvider()
    ):
        self.min_res_n = min_res_n
        self.src_location = src_location
        self.structure_format = structure_format
        self.__structure_provider = structure_provider
        super().__init__(
            src_stream=self.__get_chains(src_stream),
            src_location=SrcLocation.stream,
            structure_format=structure_format,
            structure_provider=structure_provider
        )

    def __get_chains(self, src_stream):
        chains = []
        data = pd.DataFrame(
            src_stream,
            dtype=str,
            columns=EsmProtFromStructure.COLUMNS
        ) if self.src_location == SrcLocation.stream else pd.read_csv(
            src_stream,
            header=None,
            index_col=None,
            dtype=str,
            names=EsmProtFromStructure.COLUMNS
        )
        data = data.sort_values(by=data.columns[0])
        for idx, row in data.iterrows():
            src_name = row[EsmProtFromStructure.STREAM_NAME_ATTR]
            src_structure = row[EsmProtFromStructure.STREAM_ATTR]
            item_name = row[EsmProtFromStructure.ITEM_NAME_ATTR]
            structure = self.__structure_provider.get_structure(
                src_name=src_name,
                src_structure=stringio_from_url(src_structure) if get_structure_location(src_structure) == StructureLocation.remote else src_structure,
                structure_format=self.structure_format
            )
            for ch in get_protein_chains(structure, self.min_res_n):
                chains.append((src_name, src_structure, ch, f"{item_name}.{ch}"))
        return tuple(chains)
