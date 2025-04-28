import torch
from biotite.structure import get_residues, chain_iter, filter_amino_acids
from esm.sdk.api import ESMProtein, SamplingConfig
from esm.utils.structure.protein_chain import ProteinChain

from rcsb_embedding_model.types.api_types import StreamSrc, StructureFormat
from rcsb_embedding_model.utils.model import get_aggregator_model, get_residue_model
from rcsb_embedding_model.utils.structure_parser import get_structure_from_src


class RcsbStructureEmbedding:

    MIN_RES = 10

    def __init__(self):
        self.__residue_embedding = None
        self.__aggregator_embedding = None

    def load_models(
            self,
            device: torch.device = None
    ):
        self.load_residue_embedding(device)
        self.load_aggregator_embedding(device)

    def load_residue_embedding(
            self,
            device: torch.device = None
    ):
        if not device:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__residue_embedding = _load_res_model(device)

    def load_aggregator_embedding(
            self,
            device: torch.device = None
    ):
        if not device:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__aggregator_embedding = _load_model(device)

    def structure_embedding(
            self,
            src_structure: StreamSrc,
            structure_format: StructureFormat = StructureFormat.mmcif,
            chain_id: str = None,
            assembly_id: str = None
    ):
        res_embedding = self.residue_embedding(src_structure, structure_format, chain_id, assembly_id)
        return self.aggregator_embedding(res_embedding)

    def residue_embedding(
            self,
            src_structure: StreamSrc,
            structure_format: StructureFormat = StructureFormat.mmcif,
            chain_id: str = None,
            assembly_id: str = None
    ):
        self.__check_residue_embedding()
        structure = get_structure_from_src(src_structure, structure_format, chain_id, assembly_id)
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

    def sequence_embedding(
            self,
            sequence: str
    ):
        self.__check_residue_embedding()

        if sequence.startswith(">"):
            sequence = "".join(line.strip() for line in sequence.splitlines() if not line.startswith(">"))

        if len(sequence) < RcsbStructureEmbedding.MIN_RES:
            raise ValueError(f"Sequence too short for embedding (min {RcsbStructureEmbedding.MIN_RES} residues)")

        protein = ESMProtein(sequence=sequence)
        protein_tensor = self.__residue_embedding.encode(protein)

        result = self.__residue_embedding.forward_and_sample(
            protein_tensor,
            SamplingConfig(return_per_residue_embeddings=True)
        )

        return result.per_residue_embedding

    def aggregator_embedding(
            self,
            residue_embedding: torch.Tensor
    ):
        self.__check_aggregator_embedding()
        return self.__aggregator_embedding(residue_embedding)

    def __check_residue_embedding(self):
        if self.__residue_embedding is None:
            self.load_residue_embedding()

    def __check_aggregator_embedding(self):
        if self.__aggregator_embedding is None:
            self.load_aggregator_embedding()


def _load_model(device=None):
    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    aggregator_model = get_aggregator_model(device=device)
    aggregator_model.to(device)
    aggregator_model.eval()
    return aggregator_model


def _load_res_model(device=None):
    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return get_residue_model(device)
