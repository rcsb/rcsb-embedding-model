from rcsb_embedding_model.utils.structure_parser import get_structure_from_src


class StructureProvider:

    def __init__(self):
        self.__src_name = None
        self.__structure = None

    def get_structure(
        self,
        src_name,
        src_structure,
        structure_format="mmcif",
        chain_id=None,
        assembly_id=None
    ):
        if src_name != self.__src_name:
            self.__src_name = src_name
            self.__structure = get_structure_from_src(
                src_structure=src_structure,
                structure_format=structure_format,
                assembly_id=assembly_id
            )
        if chain_id is not None:
            return self.__structure[self.__structure.chain_id == chain_id]
        return self.__structure
