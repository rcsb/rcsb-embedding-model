import os
import shutil
import unittest
import tempfile

from rcsb_embedding_model.types.api_types import StructureFormat


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
            structure_format=StructureFormat.mmcif,
            file_extension=".cif",
            index_name="test_structures",
            min_res=10,
            max_res=None,
            device="cpu"
        )

        # Verify database was created
        self.assertTrue(os.path.exists(self.__db_path))

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
            index_name="test_structures",
            top_k=5,
            threshold=None,
            output_csv=output_csv,
            min_res=10,
            max_res=None,
            device="cpu"
        )

        # Verify CSV was created
        self.assertTrue(os.path.exists(output_csv))

        # Verify CSV contains results
        with open(output_csv, 'r') as f:
            lines = f.readlines()
            # Should have header + at least some results
            self.assertGreater(len(lines), 1)

    def test_03_query_with_threshold(self):
        """Test querying with a distance threshold."""
        from rcsb_embedding_model.cli.search import query_database

        query_structure = f"{self.__test_path}/resources/pdb/1acb.cif"
        output_csv = os.path.join(self.__temp_dir, "search_results_threshold.csv")

        query_database(
            db_path=self.__db_path,
            query_structure=query_structure,
            structure_format=StructureFormat.mmcif,
            chain_id=None,
            index_name="test_structures",
            top_k=10,
            threshold=0.5,  # Apply threshold
            output_csv=output_csv,
            min_res=10,
            max_res=None,
            device="cpu"
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
            index_name="test_structures",
            top_k=5,
            threshold=None,
            output_csv=output_csv,
            min_res=10,
            max_res=None,
            device="cpu"
        )

        # Verify CSV was created
        self.assertTrue(os.path.exists(output_csv))

    def test_05_stats(self):
        """Test getting database statistics."""
        from rcsb_embedding_model.cli.search import show_statistics

        # This should run without errors
        show_statistics(
            db_path=self.__db_path,
            index_name="test_structures"
        )

    def test_06_database_builder_class(self):
        """Test EmbeddingDatabaseBuilder class directly."""
        from rcsb_embedding_model.search.database_builder import EmbeddingDatabaseBuilder

        structure_dir = f"{self.__test_path}/resources/pdb"
        output_file = os.path.join(self.__temp_dir, "test_db.pt")

        builder = EmbeddingDatabaseBuilder(
            structure_dir=structure_dir,
            structure_format=StructureFormat.mmcif,
            min_res=10,
            max_res=None,
            device="cpu"
        )

        chain_ids, embeddings = builder.build_database(
            output_path=output_file,
            file_extension=".cif"
        )

        # Verify we got some chains
        self.assertGreater(len(chain_ids), 0)
        self.assertEqual(len(chain_ids), len(embeddings))

        # Verify output file exists
        self.assertTrue(os.path.exists(output_file))

        # Test loading the database
        loaded_chain_ids, loaded_embeddings = EmbeddingDatabaseBuilder.load_database(output_file)
        self.assertEqual(chain_ids, loaded_chain_ids)
        self.assertEqual(len(embeddings), len(loaded_embeddings))

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

        query_structure = f"{self.__test_path}/resources/pdb/2uzi.cif"

        searcher = StructureSearch(
            db_path=self.__db_path,
            index_name="test_structures",
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


if __name__ == '__main__':
    unittest.main()
