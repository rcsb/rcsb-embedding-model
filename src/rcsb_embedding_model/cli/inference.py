import sys
from typing import Annotated, List

import typer

from rcsb_embedding_model.cli.args_utils import arg_devices
from rcsb_embedding_model.types.api_types import StructureFormat, Accelerator, SrcLocation, SrcProteinFrom, \
    StructureLocation, SrcAssemblyFrom, SrcTensorFrom

app = typer.Typer(
    add_completion=False
)


@app.command(
    name="residue-embedding",
    help="Calculate residue level embeddings of protein structures using ESM3. Predictions are stored as torch tensor files."
)
def residue_embedding(
        src_file: Annotated[typer.FileText, typer.Option(
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
            help='CSV file 4 (or 3) columns: Structure Name | Structure File Path or URL (switch structure-location) | Chain Id (asym_i for cif files. This field is required if src-from=chain) | Output Embedding Name.'
        )],
        output_path: Annotated[typer.FileText, typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help='Output path to store predictions. Embeddings are stored as torch tensor files.'
        )],
        src_from: Annotated[SrcProteinFrom, typer.Option(
            help='Use specific chains or all chains in a structure.'
        )] = SrcProteinFrom.chain,
        structure_location: Annotated[StructureLocation, typer.Option(
            help='Structure file location.'
        )] = StructureLocation.local,
        structure_format: Annotated[StructureFormat, typer.Option(
            help='Structure file format.'
        )] = StructureFormat.mmcif,
        min_res_n: Annotated[int, typer.Option(
            help='When using all chains in a structure, consider only chains with more than <min_res_n> residues.'
        )] = 0,
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
        src_stream=src_file,
        src_location=SrcLocation.local,
        src_from=src_from,
        structure_location=structure_location,
        structure_format=structure_format,
        min_res_n=min_res_n,
        batch_size=batch_size,
        num_workers=num_workers,
        num_nodes=num_nodes,
        accelerator=accelerator,
        devices=arg_devices(devices),
        out_path=output_path
    )


@app.command(
    name="structure-embedding",
    help="Calculate single-chain protein embeddings from structural files. Predictions are stored in a single pandas DataFrame file."
)
def structure_embedding(
        src_file: Annotated[typer.FileText, typer.Option(
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
            help='CSV file 4 (or 3) columns: Structure Name | Structure File Path or URL (switch structure-location) | Chain Id (asym_i for cif files. This field is required if src-from=chain) | Output Embedding Name.'
        )],
        output_path: Annotated[typer.FileText, typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help='Output path to store predictions. Embeddings are stored as a single DataFrame file (see out-df-name).'
        )],
        out_df_name: Annotated[str, typer.Option(
            help='File name (without extension) for storing embeddings as a pandas DataFrame pickle (.pkl). The DataFrame contains 2 columns: Id | Embedding'
        )],
        src_from: Annotated[SrcProteinFrom, typer.Option(
            help='Use specific chains or all chains in a structure.'
        )] = SrcProteinFrom.chain,
        structure_location: Annotated[StructureLocation, typer.Option(
            help='Structure file location.'
        )] = StructureLocation.local,
        structure_format: Annotated[StructureFormat, typer.Option(
            help='Structure file format.'
        )] = StructureFormat.mmcif,
        min_res_n: Annotated[int, typer.Option(
            help='When using all chains in a structure, consider only chains with more than <min_res_n> residues.'
        )] = 0,
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
        src_stream=src_file,
        src_location=SrcLocation.local,
        src_from=src_from,
        structure_location=structure_location,
        structure_format=structure_format,
        min_res_n=min_res_n,
        batch_size=batch_size,
        num_workers=num_workers,
        num_nodes=num_nodes,
        accelerator=accelerator,
        devices=arg_devices(devices),
        out_path=output_path,
        out_df_name=out_df_name
    )


@app.command(
    name="chain-embedding",
    help="Calculate single-chain protein embeddings from residue level embeddings stored as torch tensor files. Predictions are stored as csv files."
)
def chain_embedding(
        src_file: Annotated[typer.FileText, typer.Option(
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
            help='Option 1 (src-from=file) - CSV file 2 columns: Residue Embedding Torch Tensor File | Output Embedding Name. Option 2 (src-from=structure) - CSV file 3 columns: Structure Name | Structure File Path or URL (switch structure-location) | Output Embedding Name.'
        )],
        output_path: Annotated[typer.FileText, typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help='Output path to store predictions. Embeddings are stored as csv files.'
        )],
        res_embedding_location: Annotated[typer.FileText, typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help='Path where residue level embeddings are located. This argument is required if src-from=structure.'
        )] = None,
        src_from: Annotated[SrcTensorFrom, typer.Option(
            help='Use file names or all chains in a structure.'
        )] = SrcTensorFrom.file,
        structure_location: Annotated[StructureLocation, typer.Option(
            help='Structure file location.'
        )] = StructureLocation.local,
        structure_format: Annotated[StructureFormat, typer.Option(
            help='Structure file format.'
        )] = StructureFormat.mmcif,
        min_res_n: Annotated[int, typer.Option(
            help='When using all chains in a structure, consider only chains with more than <min_res_n> residues.'
        )] = 0,
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
        src_stream=src_file,
        res_embedding_location=res_embedding_location,
        src_location=SrcLocation.local,
        src_from=src_from,
        structure_location=structure_location,
        structure_format=structure_format,
        min_res_n=min_res_n,
        batch_size=batch_size,
        num_workers=num_workers,
        num_nodes=num_nodes,
        accelerator=accelerator,
        devices=arg_devices(devices),
        out_path=output_path
    )

@app.command(
    name="assembly-embedding",
    help="Calculate assembly embeddings from residue level embeddings stored as torch tensor files. Predictions are stored as csv files."
)
def assembly_embedding(
        src_file: Annotated[typer.FileText, typer.Option(
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
            help='CSV file 4 columns: Structure Name | Structure File Path or URL (switch structure-location) | Assembly Id | Output embedding name.'
        )],
        res_embedding_location: Annotated[typer.FileText, typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help='Path where residue level embeddings for single chains are located.'
        )],
        output_path: Annotated[typer.FileText, typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help='Output path to store predictions. Embeddings are stored as csv files.'
        )],
        src_from: Annotated[SrcAssemblyFrom, typer.Option(
            help='Use specific assembly or all assemblies in a structure.'
        )] = SrcAssemblyFrom.assembly,
        structure_location: Annotated[StructureLocation, typer.Option(
            help='Structure file location.'
        )] = StructureLocation.local,
        structure_format: Annotated[StructureFormat, typer.Option(
            help='Structure file format.'
        )] = StructureFormat.mmcif,
        min_res_n: Annotated[int, typer.Option(
            help='Consider only assembly chains with more than <min_res_n> residues.'
        )] = 0,
        max_res_n: Annotated[int, typer.Option(
            help='Stop adding assembly chains when number of residues is greater than <max_res_n> residues.'
        )] = sys.maxsize,
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
    from rcsb_embedding_model.inference.assembly_inferece import predict
    predict(
        src_stream=src_file,
        res_embedding_location=res_embedding_location,
        src_location=SrcLocation.local,
        src_from=src_from,
        structure_location=structure_location,
        structure_format=structure_format,
        min_res_n=min_res_n,
        max_res_n=max_res_n,
        batch_size=batch_size,
        num_workers=num_workers,
        num_nodes=num_nodes,
        accelerator=accelerator,
        devices=arg_devices(devices),
        out_path=output_path
    )


if __name__ == "__main__":
    app()
