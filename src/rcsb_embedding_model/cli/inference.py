from typing import Annotated, List

import typer

from rcsb_embedding_model.cli.args_utils import arg_devices
from rcsb_embedding_model.types.api_types import SrcFormat, Accelerator, SrcLocation

app = typer.Typer(
    add_completion=False
)


@app.command(
    name="residue-embedding",
    help="Calculate residue level embeddings of protein structures using ESM3."
)
def residue_embedding(
        src_file: Annotated[typer.FileText, typer.Option(
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
            help='CSV file 3 columns: Structure File Path | Chain Id (asym_i for cif files) | Output file name.'
        )],
        output_path: Annotated[typer.FileText, typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help='Output path to store predictions.'
        )],
        src_location: Annotated[SrcLocation, typer.Option(
            help='Source input location.'
        )] = SrcLocation.local,
        src_format: Annotated[SrcFormat, typer.Option(
            help='Structure file format.'
        )] = SrcFormat.mmcif,
        batch_size: Annotated[int, typer.Option(
            help='Number of samples processed together in one iteration.'
        )] = 1,
        num_workers: Annotated[int, typer.Option(
            help='Number of subprocesses to use for data loading.'
        )] = 0,
        num_nodes: Annotated[int, typer.Option(
            help='Number of nodes to use for inference.'
        )] = 1,
        accelerator: Annotated[Accelerator, typer.Option(
            help='Device used for inference.'
        )] = Accelerator.auto,
        devices: Annotated[List[str], typer.Option(
            help='The devices to use. Can be set to a positive number or "auto". Repeat this argument to indicate multiple indices of devices. "auto" for automatic selection based on the chosen accelerator.'
        )] = tuple(['auto'])
):
    from rcsb_embedding_model.inference.esm_inference import predict
    predict(
        csv_file=src_file,
        src_location=src_location,
        src_format=src_format,
        batch_size=batch_size,
        num_workers=num_workers,
        num_nodes=num_nodes,
        accelerator=accelerator,
        devices=arg_devices(devices),
        out_path=output_path
    )


@app.command(
    name="structure-embedding",
    help="Calculate single-chain protein embeddings from structural files. Predictions are stored in a single pandas data-frame file."
)
def structure_embedding(
        src_file: Annotated[typer.FileText, typer.Option(
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
            help='CSV file 3 columns: Structure File Path | Chain Id (asym_i for cif files) | Output file name.'
        )],
        output_path: Annotated[typer.FileText, typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help='Output path to store predictions.'
        )],
        out_df_id: Annotated[str, typer.Option(
            help='File name to store predicted embeddings.'
        )],
        src_location: Annotated[SrcLocation, typer.Option(
            help='Source input location.'
        )] = SrcLocation.local,
        src_format: Annotated[SrcFormat, typer.Option(
            help='Structure file format.'
        )] = SrcFormat.mmcif,
        batch_size: Annotated[int, typer.Option(
            help='Number of samples processed together in one iteration.'
        )] = 1,
        num_workers: Annotated[int, typer.Option(
            help='Number of subprocesses to use for data loading.'
        )] = 0,
        num_nodes: Annotated[int, typer.Option(
            help='Number of nodes to use for inference.'
        )] = 1,
        accelerator: Annotated[Accelerator, typer.Option(
            help='Device used for inference.'
        )] = Accelerator.auto,
        devices: Annotated[List[str], typer.Option(
            help='The devices to use. Can be set to a positive number or "auto". Repeat this argument to indicate multiple indices of devices. "auto" for automatic selection based on the chosen accelerator.'
        )] = tuple(['auto'])
):
    from rcsb_embedding_model.inference.structure_inference import predict
    predict(
        csv_file=src_file,
        src_location=src_location,
        src_format=src_format,
        batch_size=batch_size,
        num_workers=num_workers,
        num_nodes=num_nodes,
        accelerator=accelerator,
        devices=arg_devices(devices),
        out_path=output_path,
        out_df_id=out_df_id
    )


@app.command(
    name="chain-embedding",
    help="Calculate single-chain protein embeddings from residue level embeddings stored as torch tensor files."
)
def chain_embedding(
        src_file: Annotated[typer.FileText, typer.Option(
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
            help='CSV file 2 columns: Residue Embedding Tensor File | Output file name.'
        )],
        output_path: Annotated[typer.FileText, typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help='Output path to store predictions.'
        )],
        batch_size: Annotated[int, typer.Option(
            help='Number of samples processed together in one iteration.'
        )] = 1,
        num_workers: Annotated[int, typer.Option(
            help='Number of subprocesses to use for data loading.'
        )] = 0,
        num_nodes: Annotated[int, typer.Option(
            help='Number of nodes to use for inference.'
        )] = 1,
        accelerator: Annotated[Accelerator, typer.Option(
            help='Device used for inference.'
        )] = Accelerator.auto,
        devices: Annotated[List[str], typer.Option(
            help='The devices to use. Can be set to a positive number or "auto". Repeat this argument to indicate multiple indices of devices. "auto" for automatic selection based on the chosen accelerator.'
        )] = tuple(['auto'])
):
    from rcsb_embedding_model.inference.chain_inference import predict
    predict(
        csv_file=src_file,
        batch_size=batch_size,
        num_workers=num_workers,
        num_nodes=num_nodes,
        accelerator=accelerator,
        devices=arg_devices(devices),
        out_path=output_path
    )


if __name__ == "__main__":
    app()
