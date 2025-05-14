import logging

from esm.sdk.api import SamplingConfig
from lightning import LightningModule

from rcsb_embedding_model.utils.data import collate_seq_embeddings

logger = logging.getLogger(__name__)

class StructureModule(LightningModule):

    def __init__(
            self,
            res_model,
            aggregator_model
    ):
        super().__init__()
        logger.info(f"Using device: {self.device}")
        self.esm3 = res_model
        self.aggregator =  aggregator_model

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