from pathlib import Path
import torch

from esm.models.esm3 import ESM3
from esm.models.vqvae import StructureTokenEncoder
from esm.tokenization import TokenizerCollection, EsmSequenceTokenizer, StructureTokenizer, SecondaryStructureTokenizer, \
    SASADiscretizingTokenizer, InterProQuantizedTokenizer, ResidueAnnotationsTokenizer

from huggingface_hub import  snapshot_download

def data_root():
    path = Path(snapshot_download(repo_id="rcsb/rcsb-esm"))
    return path


def structure_encoder(device: torch.device | str = "cpu"):
    with torch.device(device):
        model = StructureTokenEncoder(
            d_model=1024, n_heads=1, v_heads=128, n_layers=2, d_out=128, n_codes=4096
        ).eval()
    state_dict = torch.load(
        data_root() / "data/weights/esm3_structure_encoder_v0.pth", map_location=device
    )
    model.load_state_dict(state_dict)
    return model



def get_model_tokenizers():

    class CustomAnnotationsTokenizer(ResidueAnnotationsTokenizer):
        def __init__(self, csv_path: str | None = None, max_annotations: int = 16):
            from esm.utils.constants import esm3 as C
            super().__init__("none", max_annotations)
            if csv_path is None:
                csv_path = str(data_root() / C.RESID_CSV)
            self.csv_path = csv_path

    return TokenizerCollection(
        sequence=EsmSequenceTokenizer(),
        structure=StructureTokenizer(),
        secondary_structure=SecondaryStructureTokenizer(kind="ss8"),
        sasa=SASADiscretizingTokenizer(),
        function=InterProQuantizedTokenizer(),
        residue_annotations=CustomAnnotationsTokenizer(),
    )


def esm_open(device: torch.device | str = "cpu"):
    with torch.device(device):
        model = ESM3(
            d_model=1536,
            n_heads=24,
            v_heads=256,
            n_layers=48,
            structure_encoder_fn=structure_encoder,
            structure_decoder_fn=lambda x: x,
            function_decoder_fn=lambda x: x,
            tokenizers=get_model_tokenizers(),
        ).eval()
    state_dict = torch.load(
        data_root() / "data/weights/esm3_sm_open_v1.pth", map_location=device
    )
    model.load_state_dict(state_dict)
    return model