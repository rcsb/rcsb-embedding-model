import sys

import pandas as pd

from rcsb_embedding_model.dataset.residue_assembly_embedding_from_tensor_file import ResidueAssemblyEmbeddingFromTensorFile
from rcsb_embedding_model.types.api_types import SrcLocation, StructureLocation, StructureFormat
from rcsb_embedding_model.utils.data import stringio_from_url
from rcsb_embedding_model.utils.structure_parser import get_assemblies
from rcsb_embedding_model.utils.structure_provider import StructureProvider


class ResidueAssemblyDatasetFromStructure(ResidueAssemblyEmbeddingFromTensorFile):

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
            max_res_n=sys.maxsize,
            structure_provider=StructureProvider()
    ):
        self.src_location = src_location
        self.structure_location = structure_location
        self.structure_format = structure_format
        self.min_res_n = min_res_n
        self.max_res_n = max_res_n
        self.__structure_provider = structure_provider
        super().__init__(
            src_stream=self.__get_assemblies(src_stream),
            res_embedding_location=res_embedding_location,
            src_location=SrcLocation.stream,
            structure_location=structure_location,
            structure_format=structure_format,
            min_res_n=min_res_n,
            max_res_n=max_res_n,
            structure_provider=structure_provider
        )

    def __get_assemblies(self, src_stream):
        assemblies = []
        for idx, row in (pd.DataFrame(
                src_stream,
                dtype=str,
                columns=ResidueAssemblyDatasetFromStructure.COLUMNS
        ) if self.src_location == SrcLocation.stream else pd.read_csv(
            src_stream,
            header=None,
            index_col=None,
            dtype=str,
            names=ResidueAssemblyDatasetFromStructure.COLUMNS
        )).iterrows():
            src_name = row[ResidueAssemblyDatasetFromStructure.STREAM_NAME_ATTR]
            src_structure = row[ResidueAssemblyDatasetFromStructure.STREAM_ATTR]
            structure = stringio_from_url(src_structure) if self.structure_location == StructureLocation.remote else src_structure
            item_name = row[ResidueAssemblyDatasetFromStructure.ITEM_NAME_ATTR]
            for assembly_id in get_assemblies(structure=structure, structure_format=self.structure_format):
                assemblies.append((src_name, src_structure, str(assembly_id), f"{item_name}.{assembly_id}"))

        return tuple(assemblies)
