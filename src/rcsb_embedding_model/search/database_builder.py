import torch
import warnings
from pathlib import Path
from typing import Optional
from tqdm import tqdm

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
            file_extension: Optional[str] = None,
            batch_size: int = 100
    ):
        """
        Build embeddings from structure files in batches (generator).

        Args:
            file_extension: File extension filter (e.g., '.cif', '.pdb'). If None, uses structure_format default
            batch_size: Number of structure files to process per batch

        Yields:
            Tuple of (chain_ids, embeddings) for each batch where chain_ids are in format "structure_name:chain_id"
        """
        if file_extension is None:
            file_extension = '.cif' if self.structure_format == StructureFormat.mmcif else '.pdb'

        if not self.structure_dir.exists():
            raise ValueError(f"Structure directory does not exist: {self.structure_dir}")

        # Collect all structure files
        structure_files = list(self.structure_dir.glob(f"*{file_extension}"))
        if not structure_files:
            raise ValueError(f"No structure files found with extension {file_extension} in {self.structure_dir}")

        total_batches = (len(structure_files) + batch_size - 1) // batch_size

        with tqdm(total=len(structure_files), desc="Processing structures", unit="file") as pbar:
            for batch_idx in range(0, len(structure_files), batch_size):
                batch_files = structure_files[batch_idx:batch_idx + batch_size]
                batch_num = (batch_idx // batch_size) + 1

                chain_ids = []
                embeddings = []

                for structure_file in batch_files:
                    try:
                        structure_name = structure_file.stem

                        # Suppress biotite warnings during structure loading
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", category=UserWarning, module="biotite")
                            warnings.filterwarnings("ignore", category=FutureWarning, module="esm")

                            # Get residue-level embeddings by chain
                            chain_residue_embeddings = self.embedder.residue_embedding_by_chain(
                                src_structure=str(structure_file),
                                structure_format=self.structure_format
                            )

                        # Add each chain to the batch (apply aggregator to get protein-level embeddings)
                        for chain_id, residue_embedding in chain_residue_embeddings.items():
                            full_chain_id = f"{structure_name}:{chain_id}"
                            chain_ids.append(full_chain_id)
                            # Apply aggregator to get protein-level embedding
                            protein_embedding = self.embedder.aggregator_embedding(residue_embedding)
                            embeddings.append(protein_embedding)

                        pbar.set_postfix({"current": structure_name, "chains": len(chain_ids)})

                    except Exception as e:
                        pbar.write(f"Error processing {structure_file.name}: {e}")
                    finally:
                        pbar.update(1)

                if chain_ids:
                    yield chain_ids, embeddings

    def build_faiss_database(
            self,
            output_db: str,
            file_extension: Optional[str] = None,
            use_gpu_index: bool = False,
            batch_size: int = 100
    ):
        """
        Build a FAISS database from structure files in batches.

        Args:
            output_db: Path to save the FAISS database (directory + prefix)
            file_extension: File extension filter (e.g., '.cif', '.pdb'). If None, uses structure_format default
            use_gpu_index: Whether to use GPU for FAISS indexing
            batch_size: Number of structure files to process per batch
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

        print("\n" + "="*80)
        print("Building embeddings and FAISS database")
        print("="*80 + "\n")

        db = FaissEmbeddingDatabase(db_path=str(db_dir), index_name=index_name)
        total_chains = 0
        first_batch = True

        for chain_ids_batch, embeddings_batch in self.build_embeddings(
                file_extension=file_extension,
                batch_size=batch_size
        ):
            if first_batch:
                # Create database with first batch
                # print(f"\nCreating FAISS database with first batch ({len(chain_ids_batch)} chains)...")
                db.create_database(chain_ids=chain_ids_batch, embeddings=embeddings_batch, use_gpu=use_gpu_index)
                first_batch = False
            else:
                # Add subsequent batches to existing database
                # print(f"\nAdding batch to database ({len(chain_ids_batch)} chains)...")
                db.add_embeddings(chain_ids=chain_ids_batch, embeddings=embeddings_batch)

            total_chains += len(chain_ids_batch)

            # Free memory after each batch
            del chain_ids_batch, embeddings_batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print("\n" + "="*80)
        print("Batch database build complete!")
        print("="*80)
        print(f"Database location: {output_db}")
        print(f"Total chains: {total_chains}")
