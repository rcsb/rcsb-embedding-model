import os
import torch
from pathlib import Path
from typing import List, Tuple

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
            device: torch.device = None,
            batch_size: int = 100
    ):
        """
        Initialize the database builder.

        Args:
            structure_dir: Directory containing structure files
            structure_format: Format of structure files (mmcif or pdb)
            min_res: Minimum residue length for chains
            max_res: Maximum residue length for structures
            device: Device to use for computation
            batch_size: Number of chains to accumulate before saving to disk
        """
        self.structure_dir = Path(structure_dir)
        self.structure_format = structure_format
        self.embedder = RcsbStructureEmbedding(min_res=min_res, max_res=max_res)
        self.embedder.load_models(device=device)
        self.batch_size = batch_size

    def build_database(
            self,
            output_path: str,
            file_extension: str = None
    ) -> Tuple[List[str], List[torch.Tensor]]:
        """
        Build embedding database from structure files.

        Args:
            output_path: Path to save the database file(s)
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

        # Setup for batch saving
        output_base = Path(output_path)
        batch_chain_ids = []
        batch_embeddings = []
        batch_num = 0
        total_chains = 0

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

                # Add each chain to the current batch
                for chain_id, residue_embedding in chain_residue_embeddings.items():
                    full_chain_id = f"{structure_name}:{chain_id}"
                    batch_chain_ids.append(full_chain_id)
                    # Apply aggregator to get protein-level embedding
                    protein_embedding = self.embedder.aggregator_embedding(residue_embedding)
                    batch_embeddings.append(protein_embedding)
                    print(f"  Added chain {chain_id} with {residue_embedding.shape[0]} residues")

                    # Save batch if it reaches batch_size
                    if len(batch_chain_ids) >= self.batch_size:
                        batch_file = f"{output_base.parent / output_base.stem}_batch_{batch_num}{output_base.suffix}"
                        self._save_batch(batch_file, batch_chain_ids, batch_embeddings)
                        total_chains += len(batch_chain_ids)
                        print(f"  Saved batch {batch_num} with {len(batch_chain_ids)} chains to {batch_file}")
                        batch_chain_ids = []
                        batch_embeddings = []
                        batch_num += 1

            except Exception as e:
                print(f"Error processing {structure_file.name}: {e}")
                continue

        # Save any remaining chains in the last batch
        if batch_chain_ids:
            batch_file = f"{output_base.parent / output_base.stem}_batch_{batch_num}{output_base.suffix}"
            self._save_batch(batch_file, batch_chain_ids, batch_embeddings)
            total_chains += len(batch_chain_ids)
            print(f"  Saved batch {batch_num} with {len(batch_chain_ids)} chains to {batch_file}")
            batch_num += 1

        if total_chains == 0:
            raise ValueError("No valid chains were processed")

        # Save metadata about the batches
        metadata = {
            'num_batches': batch_num,
            'total_chains': total_chains,
            'batch_size': self.batch_size
        }
        metadata_path = f"{output_base.parent / output_base.stem}_metadata{output_base.suffix}"
        torch.save(metadata, metadata_path)

        # Create an empty marker file at output_path for backward compatibility
        # This allows code that checks for file existence to continue working
        marker = {'batched': True, 'metadata_path': metadata_path}
        torch.save(marker, str(output_base))

        print(f"\nDatabase saved in {batch_num} batch(es)")
        print(f"Total chains: {total_chains}")
        print(f"Metadata saved to {metadata_path}")

        # Load all batches to return (for backward compatibility)
        return self.load_database(str(output_base))

    def _save_batch(self, batch_path: str, chain_ids: List[str], embeddings: List[torch.Tensor]):
        """Save a batch of embeddings to disk."""
        batch = {
            'chain_ids': chain_ids,
            'embeddings': embeddings
        }
        torch.save(batch, batch_path)

    @staticmethod
    def load_database(database_path: str) -> Tuple[List[str], List[torch.Tensor]]:
        """
        Load a previously built database.

        Args:
            database_path: Path to the database file (base path for batched databases)

        Returns:
            Tuple of (chain_ids, embeddings)
        """
        output_base = Path(database_path)
        metadata_path = f"{output_base.parent / output_base.stem}_metadata{output_base.suffix}"

        # Check if this is a batched database
        if os.path.exists(metadata_path):
            # Load from batches
            metadata = torch.load(metadata_path)
            num_batches = metadata['num_batches']

            all_chain_ids = []
            all_embeddings = []

            for batch_num in range(num_batches):
                batch_file = f"{output_base.parent / output_base.stem}_batch_{batch_num}{output_base.suffix}"
                if not os.path.exists(batch_file):
                    raise ValueError(f"Batch file does not exist: {batch_file}")

                batch = torch.load(batch_file)
                all_chain_ids.extend(batch['chain_ids'])
                all_embeddings.extend(batch['embeddings'])

            return all_chain_ids, all_embeddings
        else:
            # Legacy format: single file
            if not os.path.exists(database_path):
                raise ValueError(f"Database file does not exist: {database_path}")

            database = torch.load(database_path)
            return database['chain_ids'], database['embeddings']
