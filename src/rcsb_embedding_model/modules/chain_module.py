from lightning import LightningModule

from rcsb_embedding_model.utils.model import get_aggregator_model


class ChainModule(LightningModule):

    def __init__(
            self
    ):
        super().__init__()
        self.model = get_aggregator_model(device=self.device)

    def predict_step(self, batch, batch_idx):
        (x, x_mask), dom_id = batch
        return self.model(x, x_mask), dom_id
