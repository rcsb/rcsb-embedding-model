
import os
import shutil
import unittest

from rcsb_embedding_model.types.api_types import OutFormat, StructureFormat, Accelerator

class TestCliInference(unittest.TestCase):
    __test_path = os.path.dirname(__file__)

    def test_complete_inference(self):
        _remove_files_in_directory(f"{self.__test_path}/resources/tmp")
        from rcsb_embedding_model.cli.inference import complete_embedding
        complete_embedding(
            src_chain_file=f"{self.__test_path}/resources/src_stream/instance-complete-test.csv",
            src_assembly_file=f"{self.__test_path}/resources/src_stream/assembly-complete-test.csv",
            output_res_path=f"{self.__test_path}/resources/tmp",
            output_chain_path=f"{self.__test_path}/resources/tmp",
            output_assembly_path=f"{self.__test_path}/resources/tmp",
            output_format=OutFormat.grouped,
            output_chain_name="instance-inference",
            output_assembly_name="assembly-inference",
            structure_format=StructureFormat.bciff,
            min_res_n=0,
            batch_size_res=1,
            num_workers_res=0,
            batch_size_chain=1,
            num_workers_chain=0,
            batch_size_assembly=1,
            num_workers_assembly=0,
            num_nodes=1,
            accelerator=Accelerator.cpu
        )
        self.assertTrue(os.path.exists(f"{self.__test_path}/resources/tmp/instance-inference-0.json.gz"))
        self.assertTrue(os.path.exists(f"{self.__test_path}/resources/tmp/assembly-inference-0.json.gz"))


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