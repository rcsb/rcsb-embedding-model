import os.path
import unittest

from rcsb_embedding_model.types.api_types import SrcLocation, SrcProteinFrom, StructureFormat, \
    Accelerator, SrcAssemblyFrom


class TestRemoteInference(unittest.TestCase):

    __test_path = os.path.dirname(__file__)

    def test_esm_inference_from_structure(self):
        from rcsb_embedding_model.inference.esm_inference import predict

        esm_embeddings = predict(
            src_stream=[
                ("1acb", "https://files.rcsb.org/download/1acb.cif", "1acb"),
                ("2uzi", "https://files.rcsb.org/download/2uzi.cif", "2uzi")
            ],
            src_location=SrcLocation.stream,
            src_from=SrcProteinFrom.structure,
            structure_format=StructureFormat.mmcif,
            accelerator=Accelerator.cpu
        )

        self.assertEqual(len(esm_embeddings), 5)
        shapes = ((243, 1536), (65, 1536), (116, 1536), (106, 1536), (168, 1536))
        for idx, shape in enumerate(shapes):
            self.assertEqual(tuple(esm_embeddings[idx][0][0].shape), shape)

    def test_esm_inference_from_bcif_gz(self):
        from rcsb_embedding_model.inference.esm_inference import predict

        esm_embeddings = predict(
            src_stream=[
                ("1acb", "https://models.rcsb.org/1acb.bcif.gz", "1acb"),
                ("2uzi", "https://models.rcsb.org/2uzi.bcif.gz", "2uzi")
            ],
            src_location=SrcLocation.stream,
            src_from=SrcProteinFrom.structure,
            structure_format=StructureFormat.bciff,
            accelerator=Accelerator.cpu
        )

        self.assertEqual(len(esm_embeddings), 5)
        shapes = ((243, 1536), (65, 1536), (116, 1536), (106, 1536), (168, 1536))
        for idx, shape in enumerate(shapes):
            self.assertEqual(tuple(esm_embeddings[idx][0][0].shape), shape)


    def test_esm_inference_from_csv_bcif_gz(self):
        from rcsb_embedding_model.inference.esm_inference import predict

        esm_embeddings = predict(
            src_stream=f"{self.__test_path}/resources/src_stream/instance.csv",
            src_location=SrcLocation.file,
            src_from=SrcProteinFrom.chain,
            structure_format=StructureFormat.bciff,
            accelerator=Accelerator.cpu
        )

        self.assertEqual(len(esm_embeddings), 2)
        shapes = ((243, 1536), (116, 1536))
        for idx, shape in enumerate(shapes):
            self.assertEqual(tuple(esm_embeddings[idx][0][0].shape), shape)


    def test_esm_inference_from_cif_gz(self):
        from rcsb_embedding_model.inference.esm_inference import predict

        esm_embeddings = predict(
            src_stream=[
                ("1acb", "https://files.rcsb.org/download/1acb.cif.gz", "1acb"),
                ("2uzi", "https://files.rcsb.org/download/2uzi.cif.gz", "2uzi")
            ],
            src_location=SrcLocation.stream,
            src_from=SrcProteinFrom.structure,
            structure_format=StructureFormat.mmcif,
            accelerator=Accelerator.cpu
        )

        self.assertEqual(len(esm_embeddings), 5)
        shapes = ((243, 1536), (65, 1536), (116, 1536), (106, 1536), (168, 1536))
        for idx, shape in enumerate(shapes):
            self.assertEqual(tuple(esm_embeddings[idx][0][0].shape), shape)


    def test_assembly_inference_from_structure(self):
        from rcsb_embedding_model.inference.assembly_inferece import predict

        assembly_embeddings = predict(
            src_stream=[
                ("1acb", "https://files.rcsb.org/download/1acb.cif", "1acb"),
                ("2uzi", "https://files.rcsb.org/download/2uzi.cif", "2uzi")
            ],
            res_embedding_location=f"{self.__test_path}/resources/embeddings",
            src_location=SrcLocation.stream,
            src_from=SrcAssemblyFrom.structure,
            structure_format=StructureFormat.mmcif,
            accelerator=Accelerator.cpu
        )

        self.assertEqual(len(assembly_embeddings), 2)
