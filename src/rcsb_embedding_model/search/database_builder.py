import os
import torch
from pathlib import Path
from typing import Dict, List, Tuple

from rcsb_embedding_model.rcsb_structure_embedding import RcsbStructureEmbedding
from rcsb_embedding_model.types.api_types import StructureFormat


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

    def build_database(
            self,
            output_path: str,
            file_extension: str = None
    ) -> Tuple[List[str], List[torch.Tensor]]:
        """
        Build embedding database from structure files.

        Args:
            output_path: Path to save the database file
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

        # Save the database
        database = {
            'chain_ids': chain_ids,
            'embeddings': embeddings
        }
        torch.save(database, output_path)
        print(f"\nDatabase saved to {output_path}")
        print(f"Total chains: {len(chain_ids)}")

        return chain_ids, embeddings

    @staticmethod
    def load_database(database_path: str) -> Tuple[List[str], List[torch.Tensor]]:
        """
        Load a previously built database.

        Args:
            database_path: Path to the database file

        Returns:
            Tuple of (chain_ids, embeddings)
        """
        if not os.path.exists(database_path):
            raise ValueError(f"Database file does not exist: {database_path}")

        database = torch.load(database_path)
        return database['chain_ids'], database['embeddings']
