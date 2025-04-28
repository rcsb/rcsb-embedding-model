import unittest

from rcsb_embedding_model.types.api_types import SrcLocation, SrcProteinFrom, StructureLocation, StructureFormat, \
    Accelerator


class TestRemoteInference(unittest.TestCase):

    def test_esm_inference_from_structure(self):
        from rcsb_embedding_model.inference.esm_inference import predict

        esm_embeddings = predict(
            src_stream=[
                ("3jce", "https://files.rcsb.org/download/3jce.cif", "3jce"),
                ("9qxy", "https://files.rcsb.org/download/9qxy.cif", "9qxy")
            ],
            src_location=SrcLocation.stream,
            src_from=SrcProteinFrom.structure,
            structure_location=StructureLocation.remote,
            structure_format=StructureFormat.mmcif,
            accelerator=Accelerator.cpu
        )

        self.assertEqual(len(esm_embeddings), 52)
        shapes = ((208, 1536), (207, 1536), (152, 1536), (104, 1536), (153, 1536), (131, 1536), (129, 1536), (100, 1536), (119, 1536), (125, 1536), (116, 1536), (102, 1536), (90, 1536), (84, 1536), (82, 1536), (57, 1536), (81, 1536), (87, 1536), (53, 1536), (220, 1536), (58, 1536), (52, 1536), (48, 1536), (66, 1536), (40, 1536), (236, 1536), (272, 1536), (211, 1536), (203, 1536), (179, 1536), (178, 1536), (151, 1536), (143, 1536), (144, 1536), (124, 1536), (145, 1536), (138, 1536), (122, 1536), (118, 1536), (116, 1536), (119, 1536), (105, 1536), (112, 1536), (95, 1536), (104, 1536), (96, 1536), (81, 1536), (79, 1536), (65, 1536), (60, 1536), (588, 1536), (208, 1536))
        for idx, shape in enumerate(shapes):
            self.assertEqual(tuple(esm_embeddings[idx][0][0].shape), shape)

