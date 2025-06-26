import os
import unittest

from rcsb_embedding_model.types.api_types import Accelerator, SrcProteinFrom, SrcLocation, \
    StructureFormat, SrcAssemblyFrom, SrcTensorFrom


class TestInference(unittest.TestCase):

    __test_path = os.path.dirname(__file__)

    def test_esm_inference_from_chain(self):
        from rcsb_embedding_model.inference.esm_inference import predict
        esm_embeddings = predict(
            src_stream=[
                ("1acb", f"{self.__test_path}/resources/pdb/1acb.cif", "A", "1acb.A"),
                ("2uzi", f"{self.__test_path}/resources/pdb/2uzi.cif", "A", "2uzi.A")
            ],
            src_location=SrcLocation.stream,
            src_from=SrcProteinFrom.chain,
            structure_format=StructureFormat.mmcif,
            accelerator=Accelerator.cpu
        )

        self.assertEqual(len(esm_embeddings), 2)
        # [batch_index][item_embedding,item_name][item_index]
        self.assertEqual(tuple(esm_embeddings[0][0][0].shape), (243, 1536))
        self.assertEqual(tuple(esm_embeddings[1][0][0].shape), (116, 1536))

    def test_esm_inference_from_structure(self):
        from rcsb_embedding_model.inference.esm_inference import predict

        esm_embeddings = predict(
            src_stream=[
                ("1acb", f"{self.__test_path}/resources/pdb/1acb.cif", "1acb"),
                ("2uzi", f"{self.__test_path}/resources/pdb/2uzi.cif", "2uzi")
            ],
            src_location=SrcLocation.stream,
            src_from=SrcProteinFrom.structure,
            structure_format=StructureFormat.mmcif,
            accelerator=Accelerator.cpu
        )

        self.assertEqual(len(esm_embeddings), 5)
        # [batch_index][item_embedding,item_name][item_index]
        self.assertEqual(tuple(esm_embeddings[0][0][0].shape), (243, 1536))
        self.assertEqual(tuple(esm_embeddings[1][0][0].shape), (65, 1536))
        self.assertEqual(tuple(esm_embeddings[2][0][0].shape), (116, 1536))
        self.assertEqual(tuple(esm_embeddings[3][0][0].shape), (106, 1536))
        self.assertEqual(tuple(esm_embeddings[4][0][0].shape), (168, 1536))

    def test_chain_inference_from_tensor_files(self):
        from rcsb_embedding_model.inference.chain_inference import predict
        chain_embeddings = predict(
            src_stream=[
                (f"{self.__test_path}/resources/embeddings/1acb.A.pt", "1acb.A"),
                (f"{self.__test_path}/resources/embeddings/1acb.B.pt", "1acb.B"),
                (f"{self.__test_path}/resources/embeddings/2uzi.A.pt", "2uzi.A"),
                (f"{self.__test_path}/resources/embeddings/2uzi.B.pt", "2uzi.B"),
                (f"{self.__test_path}/resources/embeddings/2uzi.C.pt", "2uzi.C")
            ],
            src_location=SrcLocation.stream,
            accelerator=Accelerator.cpu
        )

        self.assertEqual(len(chain_embeddings), 5)
        self.assertEqual(tuple(chain_embeddings[0][0][0].shape), (1536,))
        self.assertEqual(tuple(chain_embeddings[1][0][0].shape), (1536,))
        self.assertEqual(tuple(chain_embeddings[2][0][0].shape), (1536,))
        self.assertEqual(tuple(chain_embeddings[3][0][0].shape), (1536,))
        self.assertEqual(tuple(chain_embeddings[4][0][0].shape), (1536,))

    def test_chain_inference_from_structure(self):
        from rcsb_embedding_model.inference.chain_inference import predict
        chain_embeddings = predict(
            src_stream=[
                ("1acb", f"{self.__test_path}/resources/pdb/1acb.cif", "1acb"),
                ("2uzi", f"{self.__test_path}/resources/pdb/2uzi.cif", "2uzi"),
            ],
            res_embedding_location=f"{self.__test_path}/resources/embeddings",
            src_location=SrcLocation.stream,
            src_from=SrcTensorFrom.structure,
            structure_format=StructureFormat.mmcif,
            min_res_n=0,
            accelerator=Accelerator.cpu
        )

        self.assertEqual(len(chain_embeddings), 5)
        self.assertEqual(tuple(chain_embeddings[0][0][0].shape), (1536,))
        self.assertEqual(tuple(chain_embeddings[1][0][0].shape), (1536,))
        self.assertEqual(tuple(chain_embeddings[2][0][0].shape), (1536,))
        self.assertEqual(tuple(chain_embeddings[3][0][0].shape), (1536,))
        self.assertEqual(tuple(chain_embeddings[4][0][0].shape), (1536,))

    def test_structure_inference_from_chain(self):
        from rcsb_embedding_model.inference.structure_inference import predict

        chain_embeddings = predict(
            src_stream=[
                ("1acb", f"{self.__test_path}/resources/pdb/1acb.cif", "A", "1acb.A"),
                ("2uzi", f"{self.__test_path}/resources/pdb/2uzi.cif", "A", "2uzi.A")
            ],
            src_location=SrcLocation.stream,
            src_from=SrcProteinFrom.chain,
            structure_format=StructureFormat.mmcif,
            accelerator=Accelerator.cpu
        )

        self.assertEqual(len(chain_embeddings), 2)
        # [batch_index][item_embedding,item_name][item_index]
        self.assertEqual(tuple(chain_embeddings[0][0][0].shape), (1536,))
        self.assertEqual(tuple(chain_embeddings[1][0][0].shape), (1536,))

    def test_structure_inference_from_structure(self):
        from rcsb_embedding_model.inference.structure_inference import predict

        chain_embeddings = predict(
            src_stream=[
                ("1acb", f"{self.__test_path}/resources/pdb/1acb.cif", "1acb"),
                ("2uzi", f"{self.__test_path}/resources/pdb/2uzi.cif", "2uzi")
            ],
            src_location=SrcLocation.stream,
            src_from=SrcProteinFrom.structure,
            structure_format=StructureFormat.mmcif,
            accelerator=Accelerator.cpu
        )

        self.assertEqual(len(chain_embeddings), 5)
        # [batch_index][item_embedding,item_name][item_index]
        self.assertEqual(tuple(chain_embeddings[0][0][0].shape), (1536,))
        self.assertEqual(tuple(chain_embeddings[1][0][0].shape), (1536,))
        self.assertEqual(tuple(chain_embeddings[2][0][0].shape), (1536,))
        self.assertEqual(tuple(chain_embeddings[3][0][0].shape), (1536,))
        self.assertEqual(tuple(chain_embeddings[4][0][0].shape), (1536,))

    def test_assembly_inference_from_tensor_files(self):
        from rcsb_embedding_model.inference.assembly_inferece import predict

        assembly_embedding = predict(
            src_stream=[
                ("1acb", f"{self.__test_path}/resources/pdb/1acb.cif", "1", "1acb.1"),
                ("2uzi", f"{self.__test_path}/resources/pdb/2uzi.cif", "1", "2uzi.1")
            ],
            res_embedding_location=f"{self.__test_path}/resources/embeddings",
            src_location=SrcLocation.stream,
            src_from=SrcAssemblyFrom.assembly,
            accelerator=Accelerator.cpu
        )

        self.assertEqual(len(assembly_embedding), 2)
        # [batch_index][item_embedding,item_name][item_index]
        self.assertEqual(tuple(assembly_embedding[0][0][0].shape), (1536,))
        self.assertEqual(tuple(assembly_embedding[1][0][0].shape), (1536,))

    def test_assembly_inference_from_structure(self):
        from rcsb_embedding_model.inference.assembly_inferece import predict

        assembly_embedding = predict(
            src_stream=[
                ("1acb", f"{self.__test_path}/resources/pdb/1acb.cif", "1acb"),
                ("2uzi", f"{self.__test_path}/resources/pdb/2uzi.cif", "2uzi")
            ],
            res_embedding_location=f"{self.__test_path}/resources/embeddings",
            src_location=SrcLocation.stream,
            src_from=SrcAssemblyFrom.structure,
            accelerator=Accelerator.cpu
        )

        self.assertEqual(len(assembly_embedding), 2)
        # [batch_index][item_embedding,item_name][item_index]
        self.assertEqual(tuple(assembly_embedding[0][0][0].shape), (1536,))
        self.assertEqual(tuple(assembly_embedding[1][0][0].shape), (1536,))