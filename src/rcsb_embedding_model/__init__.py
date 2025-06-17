from importlib_metadata import version, PackageNotFoundError
from rcsb_embedding_model.rcsb_structure_embedding import RcsbStructureEmbedding

try:
    __version__ = version("rcsb-embedding-model")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["RcsbStructureEmbedding", "__version__"]
