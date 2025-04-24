from esm.sdk.api import SamplingConfig
from lightning import LightningModule

from rcsb_embedding_model.utils.data import collate_seq_embeddings
from rcsb_embedding_model.utils.model import get_residue_model, get_aggregator_model


class StructureModule(LightningModule):

    def __init__(
            self
    ):
        super().__init__()
        self.esm3 = get_residue_model(self.device)
        self.aggregator = get_aggregator_model(device=self.device)

    def predict_step(self, prot_batch, batch_idx):
        prot_embeddings = []
        prot_names = []
        for esm_prot, name in prot_batch:
            embeddings = self.esm3.forward_and_sample(
                self.esm3.encode(esm_prot), SamplingConfig(return_per_residue_embeddings=True)
            ).per_residue_embedding
            prot_embeddings.append(embeddings)
            prot_names.append(name)
        res_batch_embedding, res_batch_mask = collate_seq_embeddings(prot_embeddings)
        return self.aggregator(res_batch_embedding, res_batch_mask), tuple(prot_names)