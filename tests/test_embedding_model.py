import os
import unittest

from rcsb_embedding_model import RcsbStructureEmbedding
from rcsb_embedding_model.types.api_types import SrcFormat


class TestEmbeddingModel(unittest.TestCase):

    __test_path = os.path.dirname(__file__)

    def test_residue_embedding(self):

        model = RcsbStructureEmbedding()
        res_embedding = model.residue_embedding(
            src_structure=f"{self.__test_path}/resources/1acb.cif",
            src_format=SrcFormat.mmcif,
            chain_id='A'
        )
        self.assertEqual(list(res_embedding.shape), [243, 1536])

    def test_sequence_embedding(self):

        model = RcsbStructureEmbedding()
        res_embedding = model.sequence_embedding(
            sequence="CGVPAIQPVLSGLSRIVNGEEAVPGSWPWQVSLQDKTGFHFCGGSLINENWVVTAAHCGVTTSDVVVAGEFDQGSSSEKIQKLKIAKVFKNSK"
                     "YNSLTINNDITLLKLSTAASFSQTVSAVCLPSASDDFAAGTTCVTTGWGLTRYTNANTPDRLQQASLPLLSNTNCKKYWGTKIKDAMICAGAS"
                     "GVSSCMGDSGGPLVCKKNGAWTLVGIVSWGSSTCSTSTPGVYARVTALVNWVQQTLAAN"
        )
        self.assertEqual(list(res_embedding.shape), [247, 1536])

    def test_aggregator_embedding(self):

        model = RcsbStructureEmbedding()
        res_embedding = model.residue_embedding(
            src_structure=f"{self.__test_path}/resources/1acb.cif",
            src_format=SrcFormat.mmcif,
            chain_id='A'
        )
        structure_embedding = model.aggregator_embedding(
            res_embedding
        )
        self.assertEqual(list(structure_embedding.shape), [1536])
