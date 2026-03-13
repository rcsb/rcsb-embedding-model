import os
import shutil
import unittest
import tempfile

from rcsb_embedding_model.types.api_types import StructureFormat, Accelerator


class TestCliSearch(unittest.TestCase):
    __test_path = os.path.dirname(__file__)
    __temp_dir = None
    __db_path = None
    __temp_file = None

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are used by all tests."""
        # Create temporary directory for test outputs
        cls.__temp_dir = tempfile.mkdtemp()
        cls.__db_path = os.path.join(cls.__temp_dir, "test_faiss")
        cls.__temp_file = os.path.join(cls.__temp_dir, "temp_embeddings.pt")

    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures after all tests complete."""
        if cls.__temp_dir and os.path.exists(cls.__temp_dir):
            shutil.rmtree(cls.__temp_dir)

    def test_01_build_database(self):
        """Test building a database from structure files."""
        from rcsb_embedding_model.cli.search import build_database

        # Use the test PDB files
        structure_dir = f"{self.__test_path}/resources/pdb"

        build_database(
            structure_dir=structure_dir,
            output_db=self.__db_path,
            tmp_dir=self.__temp_dir,
            structure_format=StructureFormat.mmcif,
            file_extension=".cif",
            min_res=10,
            accelerator='cpu',
            use_gpu_index=False
        )

        # Verify database files were created
        from pathlib import Path
        db_path = Path(self.__db_path)
        self.assertTrue((db_path.parent / f"{db_path.name}.index").exists())
        self.assertTrue((db_path.parent / f"{db_path.name}.metadata").exists())

    def test_02_query_database(self):
        """Test querying the database with a structure."""
        from rcsb_embedding_model.cli.search import query_database

        # Use one of the test structures as the query
        query_structure = f"{self.__test_path}/resources/pdb/1acb.cif"
        output_csv = os.path.join(self.__temp_dir, "search_results.csv")

        query_database(
            db_path=self.__db_path,
            query_structure=query_structure,
            structure_format=StructureFormat.mmcif,
            chain_id=None,
            top_k=5,
            threshold=None,
            output_csv=output_csv,
            min_res=10,
            max_res=None,
            device="cpu",
            use_gpu_index=False
        )

        # Verify CSV was created
        self.assertTrue(os.path.exists(output_csv))

        # Verify CSV contains results
        with open(output_csv, 'r') as f:
            lines = f.readlines()
            # Should have header + at least some results
            self.assertGreater(len(lines), 1)

    def test_03_query_with_threshold(self):
        """Test querying with a similarity score threshold."""
        from rcsb_embedding_model.cli.search import query_database

        query_structure = f"{self.__test_path}/resources/pdb/1acb.cif"
        output_csv = os.path.join(self.__temp_dir, "search_results_threshold.csv")

        query_database(
            db_path=self.__db_path,
            query_structure=query_structure,
            structure_format=StructureFormat.mmcif,
            chain_id=None,
            top_k=10,
            threshold=0.5,  # Apply threshold
            output_csv=output_csv,
            min_res=10,
            max_res=None,
            device="cpu",
            use_gpu_index=False
        )

        # Verify CSV was created
        self.assertTrue(os.path.exists(output_csv))

    def test_04_query_specific_chain(self):
        """Test querying with a specific chain ID."""
        from rcsb_embedding_model.cli.search import query_database

        query_structure = f"{self.__test_path}/resources/pdb/1acb.cif"
        output_csv = os.path.join(self.__temp_dir, "search_results_chain.csv")

        query_database(
            db_path=self.__db_path,
            query_structure=query_structure,
            structure_format=StructureFormat.mmcif,
            chain_id="A",  # Query specific chain
            top_k=5,
            threshold=None,
            output_csv=output_csv,
            min_res=10,
            max_res=None,
            device="cpu",
            use_gpu_index=False
        )

        # Verify CSV was created
        self.assertTrue(os.path.exists(output_csv))

    def test_05_stats(self):
        """Test getting database statistics."""
        from rcsb_embedding_model.cli.search import show_statistics

        # This should run without errors
        show_statistics(
            db_path=self.__db_path
        )

    def test_06_database_builder_class(self):
        """Test EmbeddingDatabaseBuilder class directly."""
        from rcsb_embedding_model.search.database_builder import EmbeddingDatabaseBuilder
        from rcsb_embedding_model.search.faiss_database import FaissEmbeddingDatabase

        structure_dir = f"{self.__test_path}/resources/pdb"

        # Test new FAISS database building method
        output_db = os.path.join(self.__temp_dir, "test_builder_faiss")

        builder = EmbeddingDatabaseBuilder(
            structure_dir=structure_dir,
            structure_format=StructureFormat.mmcif,
            tmp_dir=self.__temp_dir,
            min_res=10,
            accelerator="cpu"
        )

        builder.build_faiss_database(
            output_db=output_db,
            devices='auto',
            file_extension=".cif",
            use_gpu_index=False
        )

        # Verify FAISS database files exist
        from pathlib import Path
        db_path = Path(output_db).parent
        index_name = Path(output_db).name
        self.assertTrue((db_path / f"{index_name}.index").exists())
        self.assertTrue((db_path / f"{index_name}.metadata").exists())

        # Test loading the FAISS database
        db = FaissEmbeddingDatabase(db_path=str(db_path), index_name=index_name)
        db.load_database()
        stats = db.get_statistics()
        self.assertGreater(stats['total_chains'], 0)

    def test_07_faiss_database_class(self):
        """Test FaissEmbeddingDatabase class directly."""
        from rcsb_embedding_model.search.faiss_database import FaissEmbeddingDatabase
        import torch

        db_path = os.path.join(self.__temp_dir, "test_chroma_direct")
        db = FaissEmbeddingDatabase(db_path=db_path, index_name="test_direct")

        # Create some dummy embeddings
        chain_ids = ["test1:A", "test2:A", "test3:B"]
        embeddings = [torch.randn(256) for _ in range(3)]

        # Create database
        db.create_database(chain_ids=chain_ids, embeddings=embeddings)

        # Load database
        db2 = FaissEmbeddingDatabase(db_path=db_path, index_name="test_direct")
        db2.load_database()

        # Get stats
        stats = db2.get_statistics()
        self.assertEqual(stats['total_chains'], 3)

        # Search
        query_embedding = torch.randn(256)
        results_ids, results_distances = db2.search(query_embedding, top_k=2)
        self.assertEqual(len(results_ids), 2)
        self.assertEqual(len(results_distances), 2)

    def test_08_structure_search_class(self):
        """Test StructureSearch class directly."""
        from rcsb_embedding_model.search.structure_search import StructureSearch
        from pathlib import Path

        query_structure = f"{self.__test_path}/resources/pdb/2uzi.cif"

        # Parse db_path into directory and index name
        db_path_obj = Path(self.__db_path)
        db_dir = db_path_obj.parent
        index_name = db_path_obj.name

        searcher = StructureSearch(
            db_path=str(db_dir),
            index_name=index_name,
            min_res=10,
            max_res=None,
            device="cpu"
        )

        # Get statistics
        stats = searcher.get_db_statistics()
        self.assertGreater(stats['total_chains'], 0)

        # Search by structure
        results = searcher.search_by_structure(
            query_structure=query_structure,
            structure_format=StructureFormat.mmcif,
            chain_id=None,
            top_k=3
        )

        # Verify we got results
        self.assertGreater(len(results), 0)
        for query_chain, (matching_ids, distances) in results.items():
            self.assertGreater(len(matching_ids), 0)
            self.assertEqual(len(matching_ids), len(distances))

        # Test export
        output_csv = os.path.join(self.__temp_dir, "structure_search_export.csv")
        searcher.export_results(results, output_csv)
        self.assertTrue(os.path.exists(output_csv))

    def test_09_residue_embedding_by_chain_filter(self):
        """Test residue_embedding_by_chain with chain_id parameter."""
        from rcsb_embedding_model.rcsb_structure_embedding import RcsbStructureEmbedding

        query_structure = f"{self.__test_path}/resources/pdb/1acb.cif"

        embedder = RcsbStructureEmbedding(min_res=10)
        embedder.load_models(device="cpu")

        # Test getting all chains
        all_chains = embedder.residue_embedding_by_chain(
            src_structure=query_structure,
            structure_format=StructureFormat.mmcif
        )
        self.assertGreater(len(all_chains), 1)  # 1acb has multiple chains
        self.assertIn("A", all_chains)

        # Test getting only chain A
        chain_a_only = embedder.residue_embedding_by_chain(
            src_structure=query_structure,
            structure_format=StructureFormat.mmcif,
            chain_id="A"
        )
        self.assertEqual(len(chain_a_only), 1)
        self.assertIn("A", chain_a_only)
        self.assertNotIn("B", chain_a_only)

        # Verify the embeddings are the same for chain A
        import torch
        self.assertTrue(torch.equal(all_chains["A"], chain_a_only["A"]))

    def test_10_query_database_with_database(self):
        """Test querying one database against another."""
        from rcsb_embedding_model.cli.search import query_database_with_database

        output_csv = os.path.join(self.__temp_dir, "database_to_database_results.csv")

        query_database_with_database(
            query_db_path=self.__db_path,
            subject_db_path=self.__db_path,
            top_k=3,
            threshold=None,
            output_csv=output_csv,
            use_gpu_index=False
        )

        self.assertTrue(os.path.exists(output_csv))

        with open(output_csv, 'r') as f:
            lines = f.readlines()
            self.assertGreater(len(lines), 1)

    def test_11_structure_search_search_by_database(self):
        """Test StructureSearch database-to-database search API."""
        from pathlib import Path
        from rcsb_embedding_model.search.structure_search import StructureSearch

        db_path_obj = Path(self.__db_path)
        db_dir = db_path_obj.parent
        index_name = db_path_obj.name

        searcher = StructureSearch(
            db_path=str(db_dir),
            index_name=index_name
        )

        results = searcher.search_by_database(
            query_db_path=str(db_dir),
            query_index_name=index_name,
            top_k=2
        )

        self.assertGreater(len(results), 0)
        for query_chain, (matching_ids, scores) in results.items():
            self.assertGreater(len(matching_ids), 0)
            self.assertEqual(len(matching_ids), len(scores))


if __name__ == '__main__':
    unittest.main()
