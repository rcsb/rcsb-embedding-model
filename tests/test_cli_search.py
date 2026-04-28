import os
import shutil
import unittest
import tempfile

from foldmatch.types.api_types import StructureFormat


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
        from foldmatch.cli.search import build_database_from_structures

        # Use the test PDB files
        structure_dir = f"{self.__test_path}/resources/pdb"

        build_database_from_structures(
            structure_folder=structure_dir,
            output_db=self.__db_path,
            tmp_embedding_folder=self.__temp_dir,
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
        from foldmatch.cli.search import query_database_from_structure

        # Use one of the test structures as the query
        query_structure = f"{self.__test_path}/resources/pdb/1acb.cif"
        output_csv = os.path.join(self.__temp_dir, "search_results.csv")

        query_database_from_structure(
            db_path=self.__db_path,
            query_structure=query_structure,
            structure_format=StructureFormat.mmcif,
            chain_id=None,
            top_k=5,
            threshold=None,
            output_csv=output_csv,
            min_res=10,
            max_res=None,
            accelerator="cpu",
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
        from foldmatch.cli.search import query_database_from_structure

        query_structure = f"{self.__test_path}/resources/pdb/1acb.cif"
        output_csv = os.path.join(self.__temp_dir, "search_results_threshold.csv")

        query_database_from_structure(
            db_path=self.__db_path,
            query_structure=query_structure,
            structure_format=StructureFormat.mmcif,
            chain_id=None,
            top_k=10,
            threshold=0.5,  # Apply threshold
            output_csv=output_csv,
            min_res=10,
            max_res=None,
            accelerator="cpu",
            use_gpu_index=False
        )

        # Verify CSV was created
        self.assertTrue(os.path.exists(output_csv))

    def test_04_query_specific_chain(self):
        """Test querying with a specific chain ID."""
        from foldmatch.cli.search import query_database_from_structure

        query_structure = f"{self.__test_path}/resources/pdb/1acb.cif"
        output_csv = os.path.join(self.__temp_dir, "search_results_chain.csv")

        query_database_from_structure(
            db_path=self.__db_path,
            query_structure=query_structure,
            structure_format=StructureFormat.mmcif,
            chain_id="A",  # Query specific chain
            top_k=5,
            threshold=None,
            output_csv=output_csv,
            min_res=10,
            max_res=None,
            accelerator="cpu",
            use_gpu_index=False
        )

        # Verify CSV was created
        self.assertTrue(os.path.exists(output_csv))

    def test_05_stats(self):
        """Test getting database statistics."""
        from foldmatch.cli.search import show_statistics

        # This should run without errors
        show_statistics(
            db_path=self.__db_path
        )

    def test_06_database_builder_class(self):
        """Test EmbeddingDatabaseBuilder class directly."""
        from foldmatch.search.database_builder import EmbeddingDatabaseBuilder
        from foldmatch.search.faiss_database import FaissEmbeddingDatabase

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
        self.assertGreater(stats['total_embeddings'], 0)

    def test_07_faiss_database_class(self):
        """Test FaissEmbeddingDatabase class directly."""
        from foldmatch.search.faiss_database import FaissEmbeddingDatabase
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
        self.assertEqual(stats['total_embeddings'], 3)

        # Search
        query_embedding = torch.randn(256)
        results_ids, results_distances = db2.search(query_embedding, top_k=2)
        self.assertEqual(len(results_ids), 2)
        self.assertEqual(len(results_distances), 2)

    def test_08_structure_search_class(self):
        """Test StructureSearch class directly."""
        from foldmatch.search.structure_search import StructureSearch
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
        self.assertGreater(stats['total_embeddings'], 0)

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
        from foldmatch.foldmatch import FoldMatch

        query_structure = f"{self.__test_path}/resources/pdb/1acb.cif"

        embedder = FoldMatch(min_res=10)
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

    def test_10_residue_embedding_by_assembly(self):
        """Test residue_embedding_by_assembly with assembly_id parameter."""
        from foldmatch.foldmatch import FoldMatch

        query_structure = f"{self.__test_path}/resources/pdb/1acb.cif"

        embedder = FoldMatch(min_res=10)
        embedder.load_models(device="cpu")

        # Test getting asymmetric unit (no assembly_id provided)
        asymmetric_unit = embedder.residue_embedding_by_assembly(
            src_structure=query_structure,
            structure_format=StructureFormat.mmcif
        )
        self.assertEqual(len(asymmetric_unit), 1)  # Should have only asymmetric unit
        self.assertIn("0", asymmetric_unit)  # Key "0" represents asymmetric unit
        self.assertGreater(len(asymmetric_unit["0"]), 1)  # 1acb has multiple chains

        # Test getting only assembly 1
        assembly_1_only = embedder.residue_embedding_by_assembly(
            src_structure=query_structure,
            structure_format=StructureFormat.mmcif,
            assembly_id="1"
        )
        self.assertEqual(len(assembly_1_only), 1)
        self.assertIn("1", assembly_1_only)

        # Verify the asymmetric unit and assembly 1 have the same residue embeddings
        import torch
        self.assertTrue(torch.equal(asymmetric_unit["0"], assembly_1_only["1"]))


    def test_11_query_database_with_database(self):
        """Test querying one database against another."""
        from foldmatch.cli.search import query_database_with_database

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

    def test_12_structure_search_search_by_database(self):
        """Test StructureSearch database-to-database search API."""
        from pathlib import Path
        from foldmatch.search.structure_search import StructureSearch

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

    def test_13_build_database_assembly(self):
        """Test building a database with assembly granularity."""
        from foldmatch.cli.search import build_database_from_structures

        # Use the test PDB files
        structure_dir = f"{self.__test_path}/resources/pdb"
        assembly_db_path = os.path.join(self.__temp_dir, "test_faiss_assembly")

        build_database_from_structures(
            structure_folder=structure_dir,
            output_db=assembly_db_path,
            tmp_embedding_folder=self.__temp_dir,
            structure_format=StructureFormat.mmcif,
            file_extension=".cif",
            min_res=10,
            accelerator='cpu',
            use_gpu_index=False,
            granularity='assembly'
        )

        # Verify database files were created
        from pathlib import Path
        db_path = Path(assembly_db_path)
        self.assertTrue((db_path.parent / f"{db_path.name}.index").exists())
        self.assertTrue((db_path.parent / f"{db_path.name}.metadata").exists())

    def test_14_query_database_assembly(self):
        """Test querying the database with assembly granularity."""
        from foldmatch.cli.search import query_database_from_structure

        # Use the assembly database built in test_13
        assembly_db_path = os.path.join(self.__temp_dir, "test_faiss_assembly")
        query_structure = f"{self.__test_path}/resources/pdb/1acb.cif"
        output_csv = os.path.join(self.__temp_dir, "search_results_assembly.csv")

        query_database_from_structure(
            db_path=assembly_db_path,
            query_structure=query_structure,
            structure_format=StructureFormat.mmcif,
            top_k=5,
            threshold=None,
            output_csv=output_csv,
            min_res=10,
            max_res=None,
            accelerator="cpu",
            use_gpu_index=False,
            granularity='assembly',
            assembly_id='1'
        )

        # Verify CSV was created
        self.assertTrue(os.path.exists(output_csv))

        # Verify CSV contains results
        with open(output_csv, 'r') as f:
            lines = f.readlines()
            # Should have header + at least some results
            self.assertGreater(len(lines), 1)


    def test_15_update_database(self):
        """Test updating an existing database with new/replacement embeddings."""
        from foldmatch.search.faiss_database import FaissEmbeddingDatabase
        from foldmatch.cli.search import update_database_from_structures
        from pathlib import Path

        # Get initial database stats (built by test_01)
        db_path_obj = Path(self.__db_path)
        db = FaissEmbeddingDatabase(db_path=str(db_path_obj.parent), index_name=db_path_obj.name)
        db.load_database()
        initial_count = len(db.chain_ids)
        initial_ids = set(db.chain_ids)
        self.assertGreater(initial_count, 0)

        # Update with the same structure files (should replace, count stays the same)
        structure_dir = f"{self.__test_path}/resources/pdb"
        update_database_from_structures(
            structure_folder=structure_dir,
            output_db=self.__db_path,
            tmp_embedding_folder=self.__temp_dir,
            structure_format=StructureFormat.mmcif,
            file_extension=".cif",
            min_res=10,
            accelerator='cpu',
            use_gpu_index=False
        )

        db2 = FaissEmbeddingDatabase(db_path=str(db_path_obj.parent), index_name=db_path_obj.name)
        db2.load_database()
        self.assertEqual(len(db2.chain_ids), initial_count)
        self.assertEqual(set(db2.chain_ids), initial_ids)

    def test_16_update_database_unit(self):
        """Test update_embeddings directly with synthetic data."""
        from foldmatch.search.faiss_database import FaissEmbeddingDatabase
        import torch

        db_path = os.path.join(self.__temp_dir, "test_update_unit")
        db = FaissEmbeddingDatabase(db_path=db_path, index_name="test_update")

        # Create initial database
        chain_ids = ["s1:A", "s2:A", "s3:B"]
        embeddings = [torch.randn(256) for _ in range(3)]
        db.create_database(chain_ids=chain_ids, embeddings=embeddings)

        # Reload and update: replace s1:A and add s4:C
        db2 = FaissEmbeddingDatabase(db_path=db_path, index_name="test_update")
        db2.load_database()
        new_chain_ids = ["s1:A", "s4:C"]
        new_embeddings = [torch.randn(256) for _ in range(2)]
        db2.update_embeddings(chain_ids=new_chain_ids, embeddings=new_embeddings)

        # Verify: 3 original - 1 replaced + 2 new = 4 total
        db3 = FaissEmbeddingDatabase(db_path=db_path, index_name="test_update")
        db3.load_database()
        self.assertEqual(len(db3.chain_ids), 4)
        self.assertIn("s1:A", db3.chain_ids)
        self.assertIn("s2:A", db3.chain_ids)
        self.assertIn("s3:B", db3.chain_ids)
        self.assertIn("s4:C", db3.chain_ids)


    def test_17_build_database_from_embeddings(self):
        """Test building a database from pre-computed embedding .pt files."""
        from foldmatch.cli.search import build_database_from_embeddings

        embedding_dir = f"{self.__test_path}/resources/embeddings"
        output_db = os.path.join(self.__temp_dir, "test_from_embeddings")

        build_database_from_embeddings(
            embedding_folder=embedding_dir,
            output_db=output_db,
            file_extension=".pt",
            use_gpu_index=False
        )

        from pathlib import Path
        from foldmatch.search.faiss_database import FaissEmbeddingDatabase

        db_path = Path(output_db)
        self.assertTrue((db_path.parent / f"{db_path.name}.index").exists())
        self.assertTrue((db_path.parent / f"{db_path.name}.metadata").exists())

        db = FaissEmbeddingDatabase(db_path=str(db_path.parent), index_name=db_path.name)
        db.load_database()
        self.assertEqual(len(db.chain_ids), 5)

    def test_18_build_database_from_fasta(self):
        """Test building a database from a FASTA file."""
        from foldmatch.cli.search import build_database_from_fasta

        fasta_file = f"{self.__test_path}/resources/fasta/test_sequences.fasta"
        output_db = os.path.join(self.__temp_dir, "test_from_fasta")

        build_database_from_fasta(
            fasta_file=fasta_file,
            output_db=output_db,
            tmp_embedding_folder=self.__temp_dir,
            accelerator='cpu',
            use_gpu_index=False
        )

        from pathlib import Path
        from foldmatch.search.faiss_database import FaissEmbeddingDatabase

        db_path = Path(output_db)
        self.assertTrue((db_path.parent / f"{db_path.name}.index").exists())
        self.assertTrue((db_path.parent / f"{db_path.name}.metadata").exists())

        db = FaissEmbeddingDatabase(db_path=str(db_path.parent), index_name=db_path.name)
        db.load_database()
        self.assertEqual(len(db.chain_ids), 2)

    def test_19_update_database_from_embeddings(self):
        """Test updating an existing database with pre-computed embedding files."""
        from foldmatch.cli.search import build_database_from_embeddings, update_database_from_embeddings
        from foldmatch.search.faiss_database import FaissEmbeddingDatabase
        from pathlib import Path
        import torch

        embedding_dir = f"{self.__test_path}/resources/embeddings"
        output_db = os.path.join(self.__temp_dir, "test_update_from_embeddings")

        # Build initial database from .pt files (5 embeddings)
        build_database_from_embeddings(
            embedding_folder=embedding_dir,
            output_db=output_db,
            file_extension=".pt",
            use_gpu_index=False
        )

        db_path = Path(output_db)
        db = FaissEmbeddingDatabase(db_path=str(db_path.parent), index_name=db_path.name)
        db.load_database()
        initial_count = len(db.chain_ids)
        self.assertEqual(initial_count, 5)

        # Create a temp dir with new embeddings: one replacing an existing ID, one new
        update_dir = os.path.join(self.__temp_dir, "update_embeddings")
        os.makedirs(update_dir, exist_ok=True)
        torch.save(torch.randn(1536), os.path.join(update_dir, "1acb.A.pt"))  # replacement
        torch.save(torch.randn(1536), os.path.join(update_dir, "new_entry.pt"))  # new

        update_database_from_embeddings(
            embedding_folder=update_dir,
            output_db=output_db,
            file_extension=".pt",
            use_gpu_index=False
        )

        db2 = FaissEmbeddingDatabase(db_path=str(db_path.parent), index_name=db_path.name)
        db2.load_database()
        # 5 original - 1 replaced + 2 new = 6 total
        self.assertEqual(len(db2.chain_ids), 6)
        self.assertIn("1acb.A", db2.chain_ids)
        self.assertIn("new_entry", db2.chain_ids)

    def test_20_update_database_from_fasta(self):
        """Test updating an existing database with embeddings from a FASTA file."""
        from foldmatch.cli.search import build_database_from_embeddings, update_database_from_fasta
        from foldmatch.search.faiss_database import FaissEmbeddingDatabase
        from pathlib import Path

        embedding_dir = f"{self.__test_path}/resources/embeddings"
        output_db = os.path.join(self.__temp_dir, "test_update_from_fasta")

        # Build initial database from .pt embedding files (5 embeddings)
        build_database_from_embeddings(
            embedding_folder=embedding_dir,
            output_db=output_db,
            file_extension=".pt",
            use_gpu_index=False
        )

        db_path = Path(output_db)
        db = FaissEmbeddingDatabase(db_path=str(db_path.parent), index_name=db_path.name)
        db.load_database()
        initial_ids = set(db.chain_ids)
        self.assertEqual(len(initial_ids), 5)

        # Update with FASTA sequences (1acb_E, 2uzi_A - new IDs not in the initial DB)
        fasta_file = f"{self.__test_path}/resources/fasta/test_sequences.fasta"
        update_database_from_fasta(
            fasta_file=fasta_file,
            output_db=output_db,
            tmp_embedding_folder=self.__temp_dir,
            accelerator='cpu',
            use_gpu_index=False
        )

        db2 = FaissEmbeddingDatabase(db_path=str(db_path.parent), index_name=db_path.name)
        db2.load_database()
        # 5 original + 2 new from FASTA = 7 total
        self.assertEqual(len(db2.chain_ids), 7)
        self.assertIn("1acb_E", db2.chain_ids)
        self.assertIn("2uzi_A", db2.chain_ids)


    def test_21_query_database_from_embedding(self):
        """Test querying the database with a pre-computed embedding file."""
        from foldmatch.cli.search import build_database_from_embeddings, query_database_from_embedding

        embedding_dir = f"{self.__test_path}/resources/embeddings"
        subject_db = os.path.join(self.__temp_dir, "test_query_from_emb_db")
        build_database_from_embeddings(
            embedding_folder=embedding_dir,
            output_db=subject_db,
            file_extension=".pt",
            use_gpu_index=False
        )

        # Use one of the pre-computed .pt chain embedding files as the query
        embedding_file = f"{self.__test_path}/resources/embeddings/1acb.A.pt"
        output_csv = os.path.join(self.__temp_dir, "query_from_embedding_results.csv")

        query_database_from_embedding(
            db_path=subject_db,
            embedding_file=embedding_file,
            top_k=5,
            threshold=None,
            output_csv=output_csv,
            use_gpu_index=False
        )

        self.assertTrue(os.path.exists(output_csv))
        with open(output_csv, 'r') as f:
            lines = f.readlines()
            # Header + at least one result
            self.assertGreater(len(lines), 1)
            # Query ID should be the filename stem
            self.assertIn("1acb.A", lines[1])

    def test_22_query_database_from_fasta(self):
        """Test querying the database with sequences from a FASTA file."""
        from foldmatch.cli.search import build_database_from_embeddings, query_database_from_fasta

        embedding_dir = f"{self.__test_path}/resources/embeddings"
        subject_db = os.path.join(self.__temp_dir, "test_query_from_fasta_db")
        build_database_from_embeddings(
            embedding_folder=embedding_dir,
            output_db=subject_db,
            file_extension=".pt",
            use_gpu_index=False
        )

        fasta_file = f"{self.__test_path}/resources/fasta/test_sequences.fasta"
        output_csv = os.path.join(self.__temp_dir, "query_from_fasta_results.csv")

        query_database_from_fasta(
            db_path=subject_db,
            fasta_file=fasta_file,
            tmp_embedding_folder=self.__temp_dir,
            top_k=5,
            threshold=None,
            output_csv=output_csv,
            accelerator='cpu',
            use_gpu_index=False
        )

        self.assertTrue(os.path.exists(output_csv))
        with open(output_csv, 'r') as f:
            lines = f.readlines()
            # Header + results for 2 queries (1acb_E, 2uzi_A)
            self.assertGreater(len(lines), 2)
            query_ids = {line.split(',')[0] for line in lines[1:]}
            self.assertIn("1acb_E", query_ids)
            self.assertIn("2uzi_A", query_ids)


if __name__ == '__main__':
    unittest.main()
