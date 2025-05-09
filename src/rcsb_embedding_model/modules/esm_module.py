from esm.sdk.api import SamplingConfig
from lightning import LightningModule

from rcsb_embedding_model.utils.model import get_residue_model


class EsmModule(LightningModule):

    def __init__(
            self
    ):
        super().__init__()
        self.esm3 = get_residue_model(self.device)

    def predict_step(self, prot_batch, batch_idx):
        return tuple([self.__compute_embeddings(esm_prot) for esm_prot, name in prot_batch]), tuple([name for esm_prot, name in prot_batch])

    def __compute_embeddings(self, esm_prot):
        return self.esm3.forward_and_sample(
            self.esm3.encode(esm_prot), SamplingConfig(return_per_residue_embeddings=True)
        ).per_residue_embedding