import torch
from pathlib import Path
from typing import List, Tuple, Optional

from rcsb_embedding_model.rcsb_structure_embedding import RcsbStructureEmbedding
from rcsb_embedding_model.types.api_types import StructureFormat
from rcsb_embedding_model.search.faiss_database import FaissEmbeddingDatabase


class EmbeddingDatabaseBuilder:
    """Build a database of structure chain embeddings from a directory of structure files."""

    def __init__(
            self,
            structure_dir: str,
            structure_format: StructureFormat = StructureFormat.mmcif,
            min_res: int = 10,
            max_res: int = None,
            device: torch.device = None
    ):
        """
        Initialize the database builder.

        Args:
            structure_dir: Directory containing structure files
            structure_format: Format of structure files (mmcif or pdb)
            min_res: Minimum residue length for chains
            max_res: Maximum residue length for structures
            device: Device to use for computation
        """
        self.structure_dir = Path(structure_dir)
        self.structure_format = structure_format
        self.embedder = RcsbStructureEmbedding(min_res=min_res, max_res=max_res)
        self.embedder.load_models(device=device)

    def build_embeddings(
            self,
            file_extension: Optional[str] = None
    ) -> Tuple[List[str], List[torch.Tensor]]:
        """
        Build embeddings from structure files (in-memory).

        Args:
            file_extension: File extension filter (e.g., '.cif', '.pdb'). If None, uses structure_format default

        Returns:
            Tuple of (chain_ids, embeddings) where chain_ids are in format "structure_name:chain_id"
        """
        if file_extension is None:
            file_extension = '.cif' if self.structure_format == StructureFormat.mmcif else '.pdb'

        if not self.structure_dir.exists():
            raise ValueError(f"Structure directory does not exist: {self.structure_dir}")

        # Collect all structure files
        structure_files = list(self.structure_dir.glob(f"*{file_extension}"))
        if not structure_files:
            raise ValueError(f"No structure files found with extension {file_extension} in {self.structure_dir}")

        chain_ids = []
        embeddings = []

        print(f"Processing {len(structure_files)} structure files...")

        for structure_file in structure_files:
            try:
                structure_name = structure_file.stem
                print(f"Processing {structure_name}...")

                # Get residue-level embeddings by chain
                chain_residue_embeddings = self.embedder.residue_embedding_by_chain(
                    src_structure=str(structure_file),
                    structure_format=self.structure_format
                )

                # Add each chain to the database (apply aggregator to get protein-level embeddings)
                for chain_id, residue_embedding in chain_residue_embeddings.items():
                    full_chain_id = f"{structure_name}:{chain_id}"
                    chain_ids.append(full_chain_id)
                    # Apply aggregator to get protein-level embedding
                    protein_embedding = self.embedder.aggregator_embedding(residue_embedding)
                    embeddings.append(protein_embedding)
                    print(f"  Added chain {chain_id} with {residue_embedding.shape[0]} residues")

            except Exception as e:
                print(f"Error processing {structure_file.name}: {e}")
                continue

        if not chain_ids:
            raise ValueError("No valid chains were processed")

        print(f"Total chains processed: {len(chain_ids)}")
        return chain_ids, embeddings

    def build_faiss_database(
            self,
            output_db: str,
            file_extension: Optional[str] = None,
            use_gpu_index: bool = False
    ):
        """
        Build a FAISS database from structure files.

        Args:
            output_db: Path to save the FAISS database (directory + prefix)
            file_extension: File extension filter (e.g., '.cif', '.pdb'). If None, uses structure_format default
            use_gpu_index: Whether to use GPU for FAISS indexing
        """
        # Parse output_db into directory and prefix
        output_db_path = Path(output_db)
        db_dir = output_db_path.parent
        index_name = output_db_path.name

        # Ensure we have a valid directory and prefix
        if not index_name:
            index_name = "embeddings"
        if db_dir == Path('.'):
            db_dir = Path.cwd()

        # Step 1: Build embeddings from structure files
        print("\n" + "="*80)
        print("STEP 1: Building embeddings from structure files")
        print("="*80 + "\n")

        chain_ids, embeddings = self.build_embeddings(file_extension=file_extension)

        # Step 2: Create FAISS database
        print("\n" + "="*80)
        print("STEP 2: Creating FAISS database")
        print("="*80 + "\n")

        db = FaissEmbeddingDatabase(db_path=str(db_dir), index_name=index_name)
        db.create_database(chain_ids=chain_ids, embeddings=embeddings, use_gpu=use_gpu_index)

        print("\n" + "="*80)
        print("Database build complete!")
        print("="*80)
        print(f"Database location: {output_db}")
        print(f"Total chains: {len(chain_ids)}")
