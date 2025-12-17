import logging
import sys

from rcsb_embedding_model.dataset.resdiue_assembly_embedding_from_structure import ResidueAssemblyDatasetFromStructure
from rcsb_embedding_model.dataset.residue_assembly_embedding_from_tensor_file import ResidueAssemblyEmbeddingFromTensorFile
from rcsb_embedding_model.types.api_types import FileOrStreamTuple, SrcLocation, Accelerator, Devices, OptionalPath, \
    EmbeddingPath, StructureFormat, SrcAssemblyFrom, OutFormat
from rcsb_embedding_model.inference.chain_inference import predict as chain_predict


def predict(
        src_stream: FileOrStreamTuple,
        res_embedding_location: EmbeddingPath,
        src_location: SrcLocation = SrcLocation.file,
        src_from: SrcAssemblyFrom = SrcAssemblyFrom.assembly,
        structure_format: StructureFormat = StructureFormat.mmcif,
        min_res_n: int = 0,
        max_res_n: int = sys.maxsize,
        batch_size: int = 1,
        num_workers: int = 0,
        num_nodes: int = 1,
        accelerator: Accelerator = Accelerator.auto,
        devices: Devices = 'auto',
        out_format: OutFormat = OutFormat.separated,
        out_name: str = 'inference',
        out_path: OptionalPath = None
):
    logger = logging.getLogger(__name__)

    inference_set = ResidueAssemblyEmbeddingFromTensorFile(
        src_stream=src_stream,
        res_embedding_location=res_embedding_location,
        src_location=src_location,
        structure_format=structure_format,
        min_res_n=min_res_n,
        max_res_n=max_res_n
    ) if src_from == SrcAssemblyFrom.assembly else ResidueAssemblyDatasetFromStructure(
        src_stream=src_stream,
        res_embedding_location=res_embedding_location,
        src_location=src_location,
        structure_format=structure_format,
        min_res_n=min_res_n,
        max_res_n=max_res_n
    )
    logger.info(f"assembly-inference set contains {len(inference_set)} samples")

    logger.info(f"Delegating assembly-inference to chain-inference method")
    return chain_predict(
        src_stream=src_stream,
        src_location=src_location,
        batch_size=batch_size,
        num_workers=num_workers,
        num_nodes=num_nodes,
        accelerator=accelerator,
        devices=devices,
        out_format=out_format,
        out_name=out_name,
        out_path=out_path,
        inference_set=inference_set
    )