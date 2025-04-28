
from rcsb_embedding_model.dataset.residue_assembly_embedding_from_tensor_file import ResidueAssemblyEmbeddingFromTensorFile
from rcsb_embedding_model.types.api_types import FileOrStreamTuple, SrcLocation, Accelerator, Devices, OptionalPath, \
    EmbeddingPath, StructureLocation, StructureFormat
from rcsb_embedding_model.inference.chain_inference import predict as chain_predict


def predict(
        src_stream: FileOrStreamTuple,
        res_embedding_location: EmbeddingPath,
        src_location: SrcLocation = SrcLocation.local,
        structure_location: StructureLocation = StructureLocation.local,
        structure_format: StructureFormat = StructureFormat.mmcif,
        batch_size: int = 1,
        num_workers: int = 0,
        num_nodes: int = 1,
        accelerator: Accelerator = Accelerator.auto,
        devices: Devices = 'auto',
        out_path: OptionalPath = None
):
    inference_set = ResidueAssemblyEmbeddingFromTensorFile(
        src_stream=src_stream,
        res_embedding_location=res_embedding_location,
        src_location=src_location,
        structure_location=structure_location,
        structure_format=structure_format
    )

    return chain_predict(
        src_stream=src_stream,
        src_location=src_location,
        batch_size=batch_size,
        num_workers=num_workers,
        num_nodes=num_nodes,
        accelerator=accelerator,
        devices=devices,
        out_path=out_path,
        inference_set=inference_set
    )