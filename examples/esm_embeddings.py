import argparse

import torch
from biotite.structure import chain_iter, get_residues, filter_amino_acids
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.pdbx import CIFFile, get_structure, BinaryCIFFile
from esm.models.esm3 import ESM3, ESM3_OPEN_SMALL
from esm.sdk.api import ESMProtein, SamplingConfig
from esm.utils.structure.protein_chain import ProteinChain


def get_structure_from_file(file_name, format="pdb", chain_id=None):
    try:
        if format == "pdb":
            structure = PDBFile.read(file_name).get_structure(
                model=1
            )
        elif format == "mmcif":
            cif_file = CIFFile.read(file_name)
            structure = get_structure(
                cif_file,
                model=1,
                use_author_fields=False
            )
        elif format == "binarycif":
            cif_file = BinaryCIFFile.read(file_name)
            structure = get_structure(
                cif_file,
                model=1,
                use_author_fields=False
            )

        if chain_id:
            structure = structure[structure.chain_id == chain_id]
        return structure
    except:
        return None


esm3_model = ESM3.from_pretrained(ESM3_OPEN_SMALL)


def compute_embeddings(structure):
    embedding_ch = []
    for atom_ch in chain_iter(structure):
        atom_res = atom_ch[filter_amino_acids(atom_ch)]
        if len(atom_res) == 0 or len(get_residues(atom_res)[0]) < 10:
            continue
        protein_chain = ProteinChain.from_atomarray(atom_ch)
        protein = ESMProtein.from_protein_chain(protein_chain)
        protein_tensor = esm3_model.encode(protein)
        embedding_ch.append( esm3_model.forward_and_sample(
            protein_tensor, SamplingConfig(return_per_residue_embeddings=True)
        ).per_residue_embedding)
    return torch.cat(
        embedding_ch,
        dim=0
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True)
    parser.add_argument('--file_format', type=str)
    parser.add_argument('--chain', type=str)
    args = parser.parse_args()

    structure = get_structure_from_file(
        args.file,
        "pdb" if not args.file_format else args.file_format,
        chain_id=args.chain
    )

    embeddings = compute_embeddings(structure)

    print(embeddings.shape)
