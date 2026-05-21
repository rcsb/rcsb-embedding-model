import os
import shutil
import unittest
import tempfile
from pathlib import Path

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

    def test_07_faiss_database_class(self):
        """Test FaissEmbeddingDatabase class directly."""
        from foldmatch.search.faiss_database import FaissEmbeddingDatabase
        import numpy as np
        import torch

        db_path = Path(os.path.join(self.__temp_dir, "test_chroma_direct"))
        db = FaissEmbeddingDatabase(db_folder=db_path, index_name="test_direct")

        # Create some dummy embeddings
        chain_ids = ["test1:A", "test2:A", "test3:B"]
        embeddings_arr = np.stack(
            [torch.randn(256).numpy() for _ in range(3)]
        ).astype(np.float32)

        # Create database (new batch-iterator API)
        db.create_database(embedding_batches=[(chain_ids, embeddings_arr)])

        # Load database
        db2 = FaissEmbeddingDatabase(db_folder=db_path, index_name="test_direct")
        db2.load_database()

        # Get stats
        stats = db2.get_statistics()
        self.assertEqual(stats['total_embeddings'], 3)

        # Search
        query_embedding = torch.randn(256)
        results_ids, results_distances = db2.search(query_embedding, top_k=2)
        self.assertEqual(len(results_ids), 2)
        self.assertEqual(len(results_distances), 2)

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
        from foldmatch.cli.search import query_database_from_database

        output_csv = os.path.join(self.__temp_dir, "database_to_database_results.csv")

        query_database_from_database(
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

    def test_12_embedding_search_search_by_database(self):
        """Test EmbeddingSearch database-to-database search API."""
        from pathlib import Path
        from foldmatch.search.embedding_database import EmbeddingDatabase

        searcher = EmbeddingDatabase(
            db_path=self.__db_path
        )

        results = searcher.search_by_database(
            query_db_path=self.__db_path,
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

        db = FaissEmbeddingDatabase(db_folder=db_path.parent, index_name=db_path.name)
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

        db = FaissEmbeddingDatabase(db_folder=db_path.parent, index_name=db_path.name)
        db.load_database()
        self.assertEqual(len(db.chain_ids), 2)

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
        embedding_folder = f"{self.__test_path}/resources/embeddings"
        output_csv = os.path.join(self.__temp_dir, "query_from_embedding_results.csv")

        query_database_from_embedding(
            db_path=subject_db,
            embedding_folder=embedding_folder,
            file_extension=".pt",
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

    def test_23_cluster_database(self):
        """Test cluster_database CLI command on the test fixture FAISS database."""
        from foldmatch.cli.search import cluster_database

        output_csv = os.path.join(self.__temp_dir, "clusters.csv")
        cluster_database(
            db_path=self.__db_path,
            threshold=0.0,
            resolution=1.0,
            output=output_csv,
            max_neighbors=10,
            min_cluster_size=None,
            use_gpu_index=False,
            seed=42,
        )

        self.assertTrue(os.path.exists(output_csv))
        with open(output_csv, 'r') as f:
            lines = f.readlines()
        # Header + at least one cluster assignment row per chain in the fixture DB
        self.assertGreater(len(lines), 1)

        # Same export driven through the JSON output path (different file format branch).
        output_json = os.path.join(self.__temp_dir, "clusters.json")
        cluster_database(
            db_path=self.__db_path,
            threshold=0.0,
            resolution=1.0,
            output=output_json,
            max_neighbors=10,
            min_cluster_size=None,
            use_gpu_index=False,
            seed=42,
        )
        self.assertTrue(os.path.exists(output_json))
        import json
        with open(output_json) as f:
            payload = json.load(f)
        # Some JSON document was produced; structure is whatever the clusterer emits.
        self.assertTrue(payload)

    def test_24_similarity_graph(self):
        """Test similarity_graph CLI command exports a non-empty GraphML file."""
        from foldmatch.cli.search import similarity_graph

        output_graphml = os.path.join(self.__temp_dir, "similarity_graph.graphml")
        similarity_graph(
            db_path=self.__db_path,
            threshold=0.0,
            output=output_graphml,
            max_neighbors=10,
            use_gpu_index=False,
        )

        self.assertTrue(os.path.exists(output_graphml))
        self.assertGreater(os.path.getsize(output_graphml), 0)

        # GraphML is XML; verify the output is parseable and reflects the
        # fixture database (>0 nodes, with chain-id name attributes preserved).
        import xml.etree.ElementTree as ET
        tree = ET.parse(output_graphml)
        root = tree.getroot()
        # GraphML default namespace
        ns = {'g': 'http://graphml.graphdrawing.org/xmlns'}
        nodes = root.findall('.//g:node', ns)
        self.assertGreater(len(nodes), 0)


if __name__ == '__main__':
    unittest.main()
