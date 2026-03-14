import logging
import torch
import warnings
from pathlib import Path
from typing import Optional

from rcsb_embedding_model.inference.esm_inference import predict as esm_predict
from rcsb_embedding_model.inference.chain_inference import predict as chain_predict
from rcsb_embedding_model.types.api_types import StructureFormat, SrcLocation, SrcProteinFrom, Accelerator
from rcsb_embedding_model.search.faiss_database import FaissEmbeddingDatabase

logger = logging.getLogger(__name__)

class EmbeddingDatabaseBuilder:
    """Build a database of structure chain embeddings from a directory of structure files."""

    def __init__(
            self,
            structure_dir: str,
            tmp_dir: str,
            structure_format: StructureFormat = StructureFormat.mmcif,
            min_res: int = 10,
            accelerator: Accelerator = 'auto'
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
        self.tmp_dir = Path(tmp_dir)
        self.structure_format = structure_format
        self.min_res_n = min_res
        self.accelerator = accelerator

    def build_embeddings(
            self,
            file_extension: Optional[str] = None,
            devices='auto',
            batch_size_res=1,
            num_workers_res=0,
            num_nodes_res=1,
            batch_size_chain=1,
            num_workers_chain=0,
            num_nodes_chain=1,
    ):
        """
        Build embeddings from structure files in batches (generator).

        Args:
            file_extension: File extension filter (e.g., '.cif', '.pdb'). If None, uses structure_format default
            devices: Number of devices to use for inference
            batch_size_res: Number of chains to process residue embeddings per batch
            num_workers_res: Number of subprocesses to use for residue embedding data loading
            num_nodes_res: Number of nodes to use for residue embedding inference
            batch_size_chain: Number of chains to process chain embeddings per batch
            num_workers_chain: Number of subprocesses to use for chain embedding data loading
            num_nodes_chain: Number of nodes to use for chain embedding inference

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

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="esm")
            warnings.filterwarnings("ignore", category=UserWarning, module="esm")
            esm_predict(
                src_stream=[
                    (str_file.stem, str_file, str_file.stem)
                    for str_file in structure_files
                ],
                src_location=SrcLocation.stream,
                src_from=SrcProteinFrom.structure,
                structure_format=self.structure_format,
                min_res_n=self.min_res_n,
                out_path=self.tmp_dir,
                accelerator=self.accelerator,
                batch_size=batch_size_res,
                num_workers=num_workers_res,
                num_nodes=num_nodes_res,
                devices=devices
            )

        esm_embedding_files = list(self.tmp_dir.glob(f"*pt"))
        chain_embeddings = chain_predict(
            src_stream=[
                (esm_file, esm_file.stem)
                for esm_file in esm_embedding_files
            ],
            src_location=SrcLocation.stream,
            accelerator=self.accelerator,
            batch_size=batch_size_chain,
            num_workers=num_workers_chain,
            num_nodes=num_nodes_chain,
            devices=devices
        )
        return ([ch_id for _, chain_ids in chain_embeddings for ch_id in chain_ids],
                [embedding for embedding_tensor, _ in chain_embeddings for embedding in torch.split(embedding_tensor, 1, dim=0)])

    def build_faiss_database(
            self,
            output_db: str,
            file_extension: Optional[str] = None,
            use_gpu_index: bool = False,
            batch_size_res=1,
            num_workers_res=0,
            num_nodes_res=1,
            batch_size_chain=1,
            num_workers_chain=0,
            num_nodes_chain=1,
            devices='auto'
    ):
        """
        Build a FAISS database from structure files in batches.

        Args:
            output_db: Path to save the FAISS database (directory + prefix)
            file_extension: File extension filter (e.g., '.cif', '.pdb'). If None, uses structure_format default
            use_gpu_index: Whether to use GPU for FAISS indexing
            batch_size_res: Number of chains to process residue embeddings per batch
            num_workers_res: Number of subprocesses to use for residue embedding data loading
            num_nodes_res: Number of nodes to use for residue embedding inference
            num_devices_res: Number of devices to use for residue embedding inference
            batch_size_chain: Number of chains to process chain embeddings per batch
            num_workers_chain: Number of subprocesses to use for chain embedding data loading
            num_nodes_chain: Number of nodes to use for chain embedding inference
            num_devices_chain: Number of devices to use for chain embedding inference
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

        logging.info("Building embeddings and FAISS database")

        db = FaissEmbeddingDatabase(db_path=str(db_dir), index_name=index_name)

        chain_ids, embeddings = self.build_embeddings(
                file_extension=file_extension,
                devices=devices,
                batch_size_res=batch_size_res,
                num_workers_res=num_workers_res,
                num_nodes_res=num_nodes_res,
                batch_size_chain=batch_size_chain,
                num_workers_chain=num_workers_chain,
                num_nodes_chain=num_nodes_chain,
        )
        db.create_database(chain_ids=chain_ids, embeddings=embeddings, use_gpu=use_gpu_index)

        logging.info("Batch database build complete!")
        logging.info(f"Database location: {output_db}")
        logging.info(f"Total chains: {len(db.chain_ids)}")

        del chain_ids, embeddings
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
