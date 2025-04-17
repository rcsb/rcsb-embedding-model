
import torch.nn as nn
from collections import OrderedDict

from rcsb_embedding_model.model.layers import ResBlock


class ResidueEmbeddingAggregator(nn.Module):
    dropout = 0.1

    def __init__(
            self,
            input_features=1536,
            dim_feedforward=3072,
            hidden_layer=1536,
            nhead=12,
            num_layers=6,
            res_block_layers=12
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_features,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        if res_block_layers == 0:
            self.embedding = nn.Sequential(OrderedDict([
                ('norm', nn.LayerNorm(input_features)),
                ('dropout', nn.Dropout(p=self.dropout)),
                ('linear', nn.Linear(input_features, hidden_layer)),
                ('activation', nn.ReLU())
            ]))
        else:
            res_block = OrderedDict([(
                f'block{i}',
                ResBlock(input_features, hidden_layer, self.dropout)
            ) for i in range(res_block_layers)])
            res_block.update([
                ('dropout', nn.Dropout(p=self.dropout)),
                ('linear', nn.Linear(input_features, hidden_layer)),
                ('activation', nn.ReLU())
            ])
            self.embedding = nn.Sequential(res_block)

    def forward(self, x, x_mask=None):
        if x.dim() == 2:
            return self.embedding(self.transformer(x, src_key_padding_mask=x_mask).sum(dim=0))
        if x.dim() == 3:
            return self.embedding(self.transformer(x, src_key_padding_mask=x_mask).sum(dim=1))
        raise RuntimeError("Tensor dimension error. Allowed shapes (batch, sequence, residue_embeddings) or (sequence, residue_embeddings)")
