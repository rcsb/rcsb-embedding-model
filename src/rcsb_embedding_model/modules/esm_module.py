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
        prot_embeddings = []
        prot_names = []
        for esm_prot, name in prot_batch:
            embeddings = self.esm3.forward_and_sample(
                self.esm3.encode(esm_prot), SamplingConfig(return_per_residue_embeddings=True)
            ).per_residue_embedding
            prot_embeddings.append(embeddings)
            prot_names.append(name)
        return tuple(prot_embeddings), tuple(prot_names)