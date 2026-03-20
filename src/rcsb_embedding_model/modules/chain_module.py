import logging

from lightning import LightningModule

logger = logging.getLogger(__name__)

class ChainModule(LightningModule):

    def __init__(
            self,
            model
    ):
        super().__init__()
        self.aggregator = model

    def on_train_start(self):
        logger.info(f"Aggregator device: {self.device}")

    def predict_step(self, batch, batch_idx):
        (x, x_mask), dom_id = batch
        return self.aggregator(x, x_mask), dom_id
