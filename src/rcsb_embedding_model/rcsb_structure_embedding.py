import torch
from biotite.structure import get_residues, chain_iter, filter_amino_acids
from biotite.structure.io.pdb import PDBFile, get_structure as get_pdb_structure, get_assembly as get_pdb_assembly
from biotite.structure.io.pdbx import CIFFile, get_structure, get_assembly, BinaryCIFFile
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, SamplingConfig
from esm.utils.constants.models import ESM3_OPEN_SMALL
from esm.utils.structure.protein_chain import ProteinChain
from huggingface_hub import hf_hub_download

from rcsb_embedding_model.model.residue_embedding_aggregator import ResidueEmbeddingAggregator


class RcsbStructureEmbedding:

    MIN_RES = 10
    REPO_ID = "rcsb/rcsb-embedding-model"
    FILE_NAME = "rcsb-embedding-model.pt"
    VERSION = "2d71cf6"

    def __init__(self):
        self.__residue_embedding = None
        self.__aggregator_embedding = None

    def load_models(self, device=None):
        self.load_residue_embedding(device)
        self.load_aggregator_embedding(device)

    def load_residue_embedding(self, device=None):
        if not device:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__residue_embedding = _load_res_model(device)

    def load_aggregator_embedding(self, device=None):
        if not device:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__aggregator_embedding = _load_model(
            _download_model(
                RcsbStructureEmbedding.REPO_ID,
                RcsbStructureEmbedding.FILE_NAME,
                RcsbStructureEmbedding.VERSION
            ),
            device
        )

    def structure_embedding(self, structure_src, format="pdb", chain_id=None, assembly_id=None):
        res_embedding = self.residue_embedding(structure_src, format, chain_id, assembly_id)
        return self.aggregator_embedding(res_embedding)

    def residue_embedding(self, structure_src, format="pdb", chain_id=None, assembly_id=None):
        self.__check_residue_embedding()
        structure = _get_structure_from_src(structure_src, format, chain_id, assembly_id)
        embedding_ch = []
        for atom_ch in chain_iter(structure):
            atom_res = atom_ch[filter_amino_acids(atom_ch)]
            if len(atom_res) == 0 or len(get_residues(atom_res)[0]) < RcsbStructureEmbedding.MIN_RES:
                continue
            protein_chain = ProteinChain.from_atomarray(atom_ch)
            protein = ESMProtein.from_protein_chain(protein_chain)
            protein_tensor = self.__residue_embedding.encode(protein)
            embedding_ch.append(self.__residue_embedding.forward_and_sample(
                protein_tensor, SamplingConfig(return_per_residue_embeddings=True)
            ).per_residue_embedding)
        return torch.cat(
            embedding_ch,
            dim=0
        )

    def aggregator_embedding(self, residue_embedding):
        self.__check_aggregator_embedding()
        return self.__aggregator_embedding(residue_embedding)

    def __check_residue_embedding(self):
        if self.__residue_embedding is None:
            self.load_residue_embedding()

    def __check_aggregator_embedding(self):
        if self.__aggregator_embedding is None:
            self.load_aggregator_embedding()


def _get_structure_from_src(structure_src, format="pdb", chain_id=None, assembly_id=None):
    if format == "pdb":
        pdb_file = PDBFile.read(structure_src)
        structure = _get_pdb_structure(pdb_file, assembly_id)
    elif format == "mmcif":
        cif_file = CIFFile.read(structure_src)
        structure = _get_structure(cif_file, assembly_id)
    elif format == "binarycif":
        cif_file = BinaryCIFFile.read(structure_src)
        structure = _get_structure(cif_file, assembly_id)
    else:
        raise RuntimeError(f"Unknown file format {format}")

    if chain_id is not None:
        structure = structure[structure.chain_id == chain_id]

    return structure


def _get_pdb_structure(pdb_file, assembly_id = None):
    return get_pdb_structure(
        pdb_file,
        model=1
    ) if assembly_id is None else get_pdb_assembly(
        pdb_file,
        assembly_id=assembly_id,
        model=1
    )


def _get_structure(cif_file, assembly_id = None):
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


def _download_model(
        repo_id,
        filename,
        revision
):
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision
    )


def _load_model(model_path, device=None):
    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = torch.load(model_path, weights_only=True, map_location=device)
    aggregator_model = ResidueEmbeddingAggregator()
    aggregator_model.load_state_dict(weights)
    aggregator_model.to(device)
    aggregator_model.eval()
    return aggregator_model


def _load_res_model(device=None):
    return ESM3.from_pretrained(
        ESM3_OPEN_SMALL,
        device
    )
