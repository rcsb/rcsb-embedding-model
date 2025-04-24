
from biotite.structure.io.pdb import PDBFile, get_structure as get_pdb_structure, get_assembly as get_pdb_assembly
from biotite.structure.io.pdbx import CIFFile, get_structure, get_assembly, BinaryCIFFile


def get_structure_from_src(
        src_structure,
        src_format="mmcif",
        chain_id=None,
        assembly_id=None
):
    if src_format == "pdb":
        pdb_file = PDBFile.read(src_structure)
        structure = __get_pdb_structure(pdb_file, assembly_id)
    elif src_format == "mmcif":
        cif_file = CIFFile.read(src_structure)
        structure = __get_structure(cif_file, assembly_id)
    elif src_format == "binarycif":
        cif_file = BinaryCIFFile.read(src_structure)
        structure = __get_structure(cif_file, assembly_id)
    else:
        raise RuntimeError(f"Unknown file format {src_format}")

    if chain_id is not None:
        structure = structure[structure.chain_id == chain_id]

    return structure


def __get_pdb_structure(pdb_file, assembly_id=None):
    return get_pdb_structure(
        pdb_file,
        model=1
    ) if assembly_id is None else get_pdb_assembly(
        pdb_file,
        assembly_id=assembly_id,
        model=1
    )


def __get_structure(cif_file, assembly_id=None):
    return get_structure(
        cif_file,
        model=1,
        use_author_fields=False
    ) if assembly_id is None else get_assembly(
        cif_file,
        assembly_id=assembly_id,
        model=1,
        use_author_fields=False
    )
