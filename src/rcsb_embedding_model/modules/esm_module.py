import logging

from esm.sdk.api import SamplingConfig
from lightning import LightningModule

logger = logging.getLogger(__name__)

class EsmModule(LightningModule):

    def __init__(
            self,
            model
    ):
        super().__init__()
        logger.info(f"Using device: {self.device}")
        self.esm3 = model

    def predict_step(self, prot_batch, batch_idx):
        return tuple([self.__compute_embeddings(esm_prot) for esm_prot, name in prot_batch]), tuple([name for esm_prot, name in prot_batch])

    def __compute_embeddings(self, esm_prot):
        return self.esm3.forward_and_sample(
            self.esm3.encode(esm_prot), SamplingConfig(return_per_residue_embeddings=True)
        ).per_residue_embedding