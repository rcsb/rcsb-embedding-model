import torch

from huggingface_hub import hf_hub_download
from rcsb_embedding_model.model.residue_embedding_aggregator import ResidueEmbeddingAggregator
from rcsb_embedding_model.utils.esm.loaders import esm_open

REPO_ID = "rcsb/rcsb-embedding-model"
FILE_NAME = "rcsb-embedding-model.pt"
REVISION = "410606e40b1bb7968ce318c41009355c3ac32503"


def get_aggregator_model(device=None):
    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILE_NAME,
        revision=REVISION
    )
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = torch.load(model_path, weights_only=True, map_location=device)
    aggregator_model = ResidueEmbeddingAggregator()
    aggregator_model.load_state_dict(weights)
    return aggregator_model


def get_residue_model(device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return esm_open(device)

