import os
import shutil
import unittest

from foldmatch.dataset.esm_prot_from_fasta import parse_fasta
from foldmatch.types.api_types import Accelerator, OutFormat, SrcLocation, SrcTensorFrom


class TestParseFasta(unittest.TestCase):

    __test_path = os.path.dirname(__file__)

    def test_parse_fasta(self):
        sequences = parse_fasta(f"{self.__test_path}/resources/fasta/test_sequences.fasta")
        self.assertEqual(len(sequences), 2)
        self.assertEqual(sequences[0][0], "1acb_E")
        self.assertEqual(len(sequences[0][1]), 245)
        self.assertEqual(sequences[1][0], "2uzi_A")
        self.assertEqual(len(sequences[1][1]), 58)

    def test_parse_fasta_sequence_content(self):
        sequences = parse_fasta(f"{self.__test_path}/resources/fasta/test_sequences.fasta")
        self.assertTrue(sequences[0][1].startswith("CGVPAIQPVLS"))
        self.assertTrue(sequences[0][1].endswith("TLAAN"))
        self.assertTrue(sequences[1][1].startswith("RPDFCLEPP"))


class TestEsmProtFromFasta(unittest.TestCase):

    __test_path = os.path.dirname(__file__)

    def test_dataset_length(self):
        from foldmatch.dataset.esm_prot_from_fasta import EsmProtFromFasta
        dataset = EsmProtFromFasta(f"{self.__test_path}/resources/fasta/test_sequences.fasta")
        self.assertEqual(len(dataset), 2)

    def test_dataset_min_res_filter(self):
        from foldmatch.dataset.esm_prot_from_fasta import EsmProtFromFasta
        dataset = EsmProtFromFasta(
            f"{self.__test_path}/resources/fasta/test_sequences.fasta",
            min_res_n=100
        )
        self.assertEqual(len(dataset), 1)

    def test_dataset_getitem(self):
        from foldmatch.dataset.esm_prot_from_fasta import EsmProtFromFasta
        dataset = EsmProtFromFasta(f"{self.__test_path}/resources/fasta/test_sequences.fasta")
        esm_prot, name = dataset[0]
        self.assertEqual(name, "1acb_E")
        self.assertEqual(len(esm_prot), 245)


class TestSequenceInference(unittest.TestCase):

    __test_path = os.path.dirname(__file__)
    __tmp_path = os.path.join(os.path.dirname(__file__), "resources", "tmp")

    def test_sequence_residue_inference(self):
        from foldmatch.inference.sequence_inference import predict
        embeddings = predict(
            fasta_file=f"{self.__test_path}/resources/fasta/test_sequences.fasta",
            accelerator=Accelerator.cpu
        )
        self.assertEqual(len(embeddings), 2)
        # [batch_index][item_embedding,item_name][item_index]
        self.assertEqual(tuple(embeddings[0][0][0].shape), (247, 1536))
        self.assertEqual(tuple(embeddings[1][0][0].shape), (60, 1536))

    def test_sequence_residue_inference_with_output(self):
        _remove_files_in_directory(self.__tmp_path)
        from foldmatch.inference.sequence_inference import predict
        predict(
            fasta_file=f"{self.__test_path}/resources/fasta/test_sequences.fasta",
            accelerator=Accelerator.cpu,
            out_path=self.__tmp_path,
            out_format=OutFormat.separated
        )
        self.assertTrue(os.path.exists(f"{self.__tmp_path}/1acb_E.pt"))
        self.assertTrue(os.path.exists(f"{self.__tmp_path}/2uzi_A.pt"))

    def test_sequence_chain_inference(self):
        _remove_files_in_directory(self.__tmp_path)
        from foldmatch.inference.sequence_inference import predict as sequence_predict
        from foldmatch.inference.chain_inference import predict as chain_predict
        from foldmatch.cli.sequence import scan_fasta_sequences

        fasta_file = f"{self.__test_path}/resources/fasta/test_sequences.fasta"

        sequence_predict(
            fasta_file=fasta_file,
            accelerator=Accelerator.cpu,
            out_path=self.__tmp_path,
            out_format=OutFormat.separated
        )

        src_stream = scan_fasta_sequences(fasta_file, self.__tmp_path)
        chain_embeddings = chain_predict(
            src_stream=src_stream,
            src_location=SrcLocation.stream,
            src_from=SrcTensorFrom.file,
            accelerator=Accelerator.cpu
        )

        self.assertEqual(len(chain_embeddings), 2)
        self.assertEqual(tuple(chain_embeddings[0][0][0].shape), (1536,))
        self.assertEqual(tuple(chain_embeddings[1][0][0].shape), (1536,))


def _remove_files_in_directory(directory_path):
    os.makedirs(directory_path, exist_ok=True)
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")
