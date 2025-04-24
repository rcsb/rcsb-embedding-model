from typing import Annotated, List

import typer

from rcsb_embedding_model.cli.args_utils import arg_devices
from rcsb_embedding_model.types.api_types import SrcFormat, Accelerator, SrcLocation

app = typer.Typer()


@app.command(name="residue-embedding")
def residue_embedding(
        src_file: Annotated[typer.FileText, typer.Option(
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
            help='CSV file 3 columns: Structure File | Chain Id (asym_i for cif files) | Output file name.'
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
        )] = tuple(['auto']),
        output_path: Annotated[typer.FileText, typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help='Output path to store predictions.'
        )] = None
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


@app.command(name="structure-embedding")
def structure_embedding(
        src_file: Annotated[typer.FileText, typer.Option(
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
            help='CSV file 3 columns: Structure File | Chain Id (asym_i for cif files) | Output file name.'
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
        )] = tuple(['auto']),
        output_path: Annotated[typer.FileText, typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help='Output path to store predictions.'
        )] = None
):
    pass


@app.command(name="chain-embedding")
def chain_embedding(
        src_file: Annotated[typer.FileText, typer.Option(
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
            help='CSV file 2 columns: Residue Embedding Tensor File | Output file name.'
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
        )] = tuple(['auto']),
        output_path: Annotated[typer.FileText, typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help='Output path to store predictions.'
        )] = None
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
